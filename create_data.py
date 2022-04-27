import os


# 生成数据列表
def get_data_list(audio_path, list_path):
    sound_sum = 0
    audios = os.listdir(audio_path)

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w')
    f_label = open(os.path.join(list_path, 'label_list.txt'), 'w')

    for i in range(len(audios)):
        f_label.write(f'{audios[i]}\n')
        sounds = os.listdir(os.path.join(audio_path, audios[i]))
        for sound in sounds:
            sound_path = os.path.join(audio_path, audios[i], sound).replace('\\', '/')
            f_train.write('%s\t%d\n' % (sound_path, i))
            sound_sum += 1
    f_label.close()
    f_train.close()


if __name__ == '__main__':
    get_data_list('dataset/Data_MGTV', 'dataset')