import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import os
import builtins
import time
import argparse
import ResNet as RN
import PyramidNet as PYRM

# 0. main worker 정의
# main함수로 mp.spwan함수를 통해 하나의 프로세스에서 하나의 main worker 동작
def main():
    args = parser.parse_args()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size=ngpus_per_node*args.world_size
        mp.spawn(main_worker,nprocs=ngpus_per_node,args=(ngpus_per_node,args))
    else:
        main_worker(args.gpu,ngpus_per_node,args)

# 1. gpu 설정
def main_worker(gpu,ngpus_per_node, args):

    # 내용1 :gpu 설정
    print(gpu,ngpus_per_node)
    args.gpu = gpu
    global best_err1, best_err5

    # 내용1-1: gpu!=0이면 print pass
    if args.multiprocessing_distributed and args.gpu !=0:
        def print_pass(*args):
            pass
        builtins.print=print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    if args.distributed:
        if args.dist_url=='env://' and args.rank==-1:
            args.rank=int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # gpu = 0,1,2,...,ngpus_per_node-1
            print("gpu는",gpu)
            args.rank=args.rank*ngpus_per_node + gpu
        # 내용1-2: init_process_group 선언
        torch.distributed.init_process_group(backend=args.dist_backend,init_method=args.dist_url,
                                            world_size=args.world_size,rank=args.rank)
    # 내용2: model 정의
    print("=> creating model '{}'".format(args.net_type))

    if args.dataset == 'cifar100':
        numberofclass = 100
    elif args.dataset == 'cifar10':
        numberofclass = 10
    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)  # for ResNet
    elif args.net_type == 'pyramidnet':
        model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
                                args.bottleneck)
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    # 내용3: multiprocess 설정
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # when using a single GPU per process and per DDP, we need to divide tha batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers+ngpus_per_node-1)/ngpus_per_node)
            # 내용3-1: model ddp설정
            model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu])# args.gpu가 무슨 값인지 알고 싶다.
        else:
            model.cuda()
            # DDP will divide and allocate batch_size to all available GPUs if device_ids are not set
            # 만약에 device_ids를 따로 설정해주지 않으면, 가능한 모든 gpu를 기준으로 ddp가 알아서 배치사이즈와 workers를 나눠준다는 뜻.
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model=model.cuda(args.gpu)
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataparallel is supported.")
    
    # 내용4: criterion / optimizer 정의
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    
    # 내용5: 데이터 로딩
    # 내용5-1: transform 정의
    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        # 내용5-2: dataset 정의
        if args.dataset == 'cifar100':
            train_dataset = datasets.CIFAR100('../data', train=True, download=True, transform=transform_train)
            val_dataset = datasets.CIFAR100('../data', train=False, transform=transform_test)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform_train)
            val_dataset = datasets.CIFAR10('../data', train=False, transform=transform_test)
            numberofclass = 10
            
        # 내용5-3: sampler 정의 (참고: val_loader는 sampler를 사용하지 않는다.)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True,sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    # 내용 6: for문을 통한 training
    cudnn.benchmark = True
    stime = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.start_epoch, args.epochs):
        # 내용 6-1: train_sampler.set_epoch
        # In distributed mode, calling the set_eopch() method at the beggining of 
        # each epoch before creating the "dataloader" iterator is necessary to make
        # suffling work properly across multiple epochs. Otherwise, the same ordering will be always used.
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch,args)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch,args,scaler)

        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch,args)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5

        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank% ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch,
                'arch': args.net_type,
                'state_dict': model.state_dict(),
                'best_err1': best_err1,
                'best_err5': best_err5,
                'optimizer': optimizer.state_dict(),
            }, is_best,args=args)
    etime = time.time()
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        print("총 걸린시간: ",etime-stime)
        print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)