#!/bin/bash
## in/out path 
comment=("title")
len_comm=${#comment[@]}


#checkpoint="./result/20210418/model/MINANet_title_210418-153730.pth"    # Pretrain위에 다시 train한 결과
#checkpoint="./result/20210418/model/MINANet_title_210418-152154.pth"        # 저장된 모델의 (및 config) path - pretrain=True로 됨.
                                                                            # mode가 train이면 pretrained된 모델을 다시 학습
info_path="../physionet_subject_map.csv"        # 데이터 저장된 csv
#info_path="../PTBXL_subjectmap_N_MI.csv"        # 데이터 저장된 csv
#ext_path="../physionet_subject_map.csv"        # EXTERNAL validation할때 사용


mode="train"                                    # train, test, intv 
num_workers="10"                                # number of data loaders (default = 8)
gpus="0,1"   #4,5에서 0,1로 바꿈                 # used gpu (example:"0,1")


# model resnet 
# kernel_size=("3" "7") 
# loss_lambda=("0.7")
# inplanes=("16" "32" "64")
# model_output=("128" "64") # "32")
loss_lambda=("0.7")
inplanes=("64")
kernel_size=("32")
model_output=("8") #"64") # "32")



len_inplanes=${#inplanes[@]}
len_kernel_size=${#kernel_size[@]}
len_loss_lambda=${#loss_lambda[@]} 
len_model_output=${#model_output[@]} 

## model common
num_classes="2"
oversample_ratio=("0")
batch_size=("512")
num_epochs=("20")
hidden_size=("32") #"32") #"128" #12리드 합칠때 fc에서의 중간node갯수
lr=("1e-4")
dropout=("0.5")
# for focal loss
gamma=("2")
alpha=("0.5") # "0.3") # "0.7")
# for calibration
calibration=("1")

n_channel=("4")
n_split=("50")

len_oversample_ratio=${#oversample_ratio[@]}
len_batch_size=${#batch_size[@]}
len_num_epochs=${#num_epochs[@]}
len_hidden_size=${#hidden_size[@]}
len_lr=${#lr[@]}
len_dropout=${#dropout[@]}
len_gamma=${#gamma[@]}
len_alpha=${#alpha[@]}
len_calibration=${#calibration[@]} 

for ((calib=0;calib<$len_calibration;calib++)) do 
    for ((comm=0;comm<$len_comm;comm++)) do 
        for ((over=0;over<$len_oversample_ratio;over++)) do 
            for ((inp=0;inp<$len_inplanes;inp++)) do
                for ((kernel=0;kernel<$len_kernel_size;kernel++)) do 
                    for ((mout=0;mout<$len_model_output;mout++)) do  
                        for ((batch=0;batch<$len_batch_size;batch++)) do 
                            for ((ep=0;ep<$len_num_epochs;ep++)) do 
                                for ((hidden=0;hidden<$len_hidden_size;hidden++)) do 
                                    for ((learn_rate=0;learn_rate<$len_lr;learn_rate++)) do 
                                        for ((drop=0;drop<$len_dropout;drop++)) do 
                                            for ((lambda=0;lambda<$len_loss_lambda;lambda++)) do 
                                                for ((gam=0;gam<$len_gamma;gam++)) do 
                                                    for ((alph=0;alph<$len_alpha;alph++)) do 
                                                        now_date=$(date +"%y%m%d-%H%M%S")
                                                        dir=$(date +"%y%m%d")
                                                        mkdir -p ./log/$dir
                                                        tmp_comm=${comment[comm]}\_$now_date
                                                        # arguments들을 옆으로 쓰니 됨
                                                        # --checkpoint $checkpoint\ # checkpoint 쓰면 이거 넣어야댐
                                                        python ./main.py --comment $tmp_comm\
                                                        --data_path $info_path\
                                                        --mode $mode\
                                                        --num_workers $num_workers\
                                                        --gpus $gpus\
                                                        --inplanes ${inplanes[inp]}\
                                                        --kernel_size ${kernel_size[kernel]}\
                                                        --model_output ${model_output[mout]}\
                                                        --n_channel $n_channel\
                                                        --n_split $n_split\
                                                        --oversample_ratio ${oversample_ratio[over]}\
                                                        --batch_size ${batch_size[batch]}\
                                                        --num_epochs ${num_epochs[ep]}\
                                                        --hidden_size ${hidden_size[hidden]}\
                                                        --lr ${lr[learn_rate]}\
                                                        --dropout ${dropout[drop]}\
                                                        --loss_lambda ${loss_lambda[lambda]}\
                                                        --gamma ${gamma[gam]}\
                                                        --num_classes $num_classes\
                                                        --alpha ${alpha[alph]}\
                                                        --calibration ${calibration[calib]}
                                                        &>> ./log/$dir/$tmp_comm.log
                                                    done
                                                done
                                            done
                                        done #checkpoint 위치 확인
                                    done
                                done
                            done
                        done
                    done 
                done 
            done 
        done
    done
done