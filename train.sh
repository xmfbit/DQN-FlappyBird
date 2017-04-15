lr=0.00005
gamma=0.99
batch_size=32
mem_size=5000
initial_epsilon=0.2
final_epsilon=0.1
observation=100
exploration=1000
max_episode=2000
# for fine tuning, uncomment this
weight=model_best.pth.tar

python main.py --train\
               --cuda\
               --lr=$lr\
               --gamma=$gamma\
               --batch_size=$batch_size\
               --memory_size=$mem_size\
               --init_e=$initial_epsilon\
               --final_e=$final_epsilon\
               --observation=$observation\
               --exploration=$exploration\
               --max_episode=$max_episode\
               --weight=$weight   # for fine tuning, uncomment this

