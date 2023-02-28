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

# 각종 모듈 import
# torch.backends.cudnn은 딥러닝 GPU의 딥러닝 연산을 지원하는 모듈이다
# torch.multiprocessing은 멀티프로세싱을 지원하는 모듈이다.
# jobs는 직접 만든 모듈로, train을 위한 실질적인 코드를 실행시킨다.(같은 디렉토리의jobs.py확인)
import os
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from jobs import train


# 환경변수를 지정하는 sys모듈의 path저장소에 insert 함수를 사용하여 주소를 입력한다.
# 그 안의 os.path.abspath는 절대값으로 경로를 생성하는 함수이며
# 그 안의 os.path.join은 인자로 입력받은 값을 경로로 만들어 주는 함수이다. 
# 이렇게 하는 이유는 운영체제마다 경로 구분자가 다르기 때문이다.(운영체제별로 폴더 및 파일 경로를 표시할 때 \, \\, /등 다양하게 표시함)
# 그 안의 os.path.dirname은 __file__에서 디렉토리명만 추출하는 것이다.
# __file__은 train.py가 수행되고 있는 path를 출력한다.
# 결국 아래 코드는 형재 파일의 경로를 환경변수를 지정하는 것이다.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


# 해당 파일이 시작점인지 확인하는 코드
# __name__은 해당 파일의 이름을 표시하지만
# 처음 실행시키는 파일의 __name__은 __main__으로 표시된다(만약 시작점이 아니었다면 'train'이었을 것이다.)
if __name__ == '__main__':

    # train.py가 시작점이면 config 파일에서 
    # #####parse_args를 import한다.

    from configs.config import parse_args

    # 설정 받아오기
    cfg, args = parse_args()

    # 이미 생성된 cfg파일을 사용한다면(parse_arge에서 cfg_file을 불러오는지 확인한 뒤, cfg객체에 정보를 담아서 출력함)
    # cfg.output_dir안에 path를 입력한다.(무슨 용도인지는 잘 모르겠다.)
    if cfg.cfg_file is not None:
        exp_name = cfg.cfg_file.split('/')[-1].split('.')[0]
        cfg.output_dir = os.path.join('./output', exp_name)

    # cudnn.benchmark는 convolution 수행시 현재 환경에 가장 적합한 알고리즘을 선정해 수행시킨다.
    # cudnn.deterministic은 cuda에서 deterministic한 연산만 수행되도록 한다. deterministic이 무엇인지는 잘 모르겠다.
    # cudnn.benchmark= False와 cudnn.deterministic = True는 reproducibility를 위함으로 보인다.
    # (reproducibility는 같은 데이터를 사용했을 때 같은 결과가 나오도록 만드는 성질이다.)
    cudnn.benchmark = False
    cudnn.deterministic = True
    # 사용되지 않는 GPU상 cache를 정리
    torch.cuda.empty_cache()
    # 사용가능 GPU를 num_gpus에 할당한다.
    num_gpus = torch.cuda.device_count()

    # 멀티프로세싱을 지원하는 모듈, 일반적으로 코드는 순차적으로 실행되며 순차적으로 종료되는데
    # 순차적인 프로세스 실행은 가끔씩 문제가 생길 수 있다.(비정상 상황에서 종료가 되지 않아 다음 프로세스로 넘어가지 않는다든지)
    # spawn 함수는 이러한 문제를 해결해준다.(자동 종료 등)
    # 0이 아닌 상태로 종료될 경우(비정상 종료를 말하는 듯하다) 나머지 프로세스까지 종료하고 종료 원인에 대한 예외를 발생시킨다.
    # 아래 코드는 train을 spawn 함수를 통해 실행시킨다.
    # 전달되는 인자에서 train은 사용하는 함수 이름, args는 train에들어가는 인자(사용할 GPU수와 설정(cfg), 
    # nprocs는 생성할 프로세스 수, join은 모든 프로세스에 blocking join적용 적용 여부(blocking join이 뭔지는 못 찾음...)
    mp.spawn(train, args=(num_gpus, cfg), nprocs=num_gpus, join=True)

    exit(0)
