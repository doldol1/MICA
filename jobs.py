# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de

# torch.distributed는 GPU를 통해 분산 프로세스를 실행시키는데 사용함
# yaml은 각종 설정을 저장, 실행하는데 사용됨
# loguru는 logging을 하는데 사용하는 모듈(logging: log 작성)
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from loguru import logger

# 아래 세 모듈은 MICA에서 직접 제작한 모듈
from micalib.tester import Tester
from micalib.trainer import Trainer
from utils import util

# 이 파일이 있는 경로도 환경변수 지정함
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


# 실행 사전작업, rank가 어디서 받아오는지는 확인 불명
# os.environ은 환경 변수를 접근할 수 있도록 한다. 
# 'MASTER_ADDR'은 0순위의 프로세스를 호스팅할 컴퓨터의 IP주소를 의미하고
# 'MASTER_PORT'는 0순위의 프로세스를 호스팅할 컴퓨터의 사용 가능한 포트를 의미한다.
# 즉, 프로세스를 시작하는 0순위 컴퓨터는 localhost, 포트는 port로 지정한다는 뜻이다.
# dist.init_process_group는 프로세스 그룹(프로세스들의 집합)을 초기화시킨다.
# "nccl"은 백엔드 빌드의 이름(멀티프로세싱 빌드를 말하는 듯), rank는 현재 process의 rank(순서나 우선순위 의미인듯)
# world_size는 프로세스의 수, init_method는 초기화 방법
def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method="env://")


def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)

    cudnn.deterministic = True
    cudnn.benchmark = False


def test(rank, world_size, cfg, args):
    port = np.random.randint(low=0, high=2000)
    setup(rank, world_size, 12310 + port)

    deterministic(rank)

    cfg.model.testing = True
    mica = util.find_model_using_name(model_dir='micalib.models', model_name=cfg.model.name)(cfg, rank)
    tester = Tester(nfc_model=mica, config=cfg, device=rank)
    tester.render_mesh = True

    if args.test_dataset.upper() == 'STIRLING':
        tester.test_stirling(args.checkpoint)
    elif args.test_dataset.upper() == 'NOW':
        tester.test_now(args.checkpoint)
    else:
        logger.error('[TESTER] Test dataset was not specified!')

    dist.destroy_process_group()


# train.py에서 mp.spawn으로 호출하는 함수 mp.spawn에서 매개변수는 2개만(num_gpus, cfg)넣었는데 돌아가는지는 불명
def train(rank, world_size, cfg):
    port = np.random.randint(low=0, high=2000)
    # 상단 def setup 확인, 멀티프로세스 등 초기화
    setup(rank, world_size, 12310 + port)

    ### 여기부터는 분석 시간관계로 조금 빠르게 감(나중에라도 다 볼것)
    # 로그 기록, loguru는 log를 등급별로 나누어 중요한 log와 덜 중요한 log를 등급으로 나눈다.
    # 자세한 사항은 https://loguru.readthedocs.io/en/stable/api/logger.html#loguru._logger.Logger.info
    # 의 'The severity levels'을 참조
    logger.info(f'[MAIN] output_dir: {cfg.output_dir}')
    # 로그 출력 디렉토리 생성으로 보인다.
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.vis_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.val_vis_dir), exist_ok=True)

    
    with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
        # yaml.dump()는 yaml파일의 내용을 출력하는 역할인데... 왜 print문도 사용하지 않는지 의문
        yaml.dump(cfg, f, default_flow_style=False)
    # shutil.copy(cfg.cfg_file, os.path.join(cfg.output_dir, 'config.yaml'))

    deterministic(rank)

    # util.find_model_using_name()은 모델을 불러오는 역할을 한다.
    # cfg.model.name은 micalib.models에 있는 mica 파일에서 모델을 불러온다.
    nfc = util.find_model_using_name(model_dir='micalib.models', model_name=cfg.model.name)(cfg, rank)
    # trainer instance를 생성한다.nfc는 모델, cfg는 설정값, device는 rank값이다.
    trainer = Trainer(nfc_model=nfc, config=cfg, device=rank)
    trainer.fit()

    dist.destroy_process_group()
