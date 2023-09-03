# Проект ft_linear_regression
Реализация модели линейной регрессии без использования специальных библиотек для задачи прогнозирования цены автомобиля в зависимости от пробега; регуляризация; MSE, R-squared statistic; визуализация функций потерь и предсказаний цены по данной выборке

## Начало Работы

### Копирование
Для копирования файлов Проекта на локальный компьютер в папку *<your_dir_on_local_computer>* выполните:

```
    $ git clone git@github.com:dbadeev/ft_linear_regression.git <your_dir_on_local_computer>
```

### Описание файлов
* *ft_linreg.en.pdf* - текст задания
* *requirements.txt* - список библиотек, необходимый для работы 
* *data.csv* - Ффайл с данными по машинам (пробег/стоимость)  
* *train.py* - программа обучения модели  
* *predict.py* - программа, предсказывающая цену машины по ее пробегу 

<br>
<br>
  
## Запуск программ

1. После установки необходимых библиотек, сначала выполняется обучение модели на имеющихся данных запуском программы _train.py_. <br> <br>
Чтобы вывести на экран информацию о параметрах, 
из папки *<your_dir_on_local_computer>* выполните *train.py* с ключом _-h_:

```
$ python3 train.py -h          
usage: train.py [-h] [--path PATH] [--loss_control LOSS_CONTROL] [--epochs EPOCHS] [--learning_rate ETA] [--loss_graphics]
                [--predict_data] [--animation] [--debug] [--quality]

options:
  -h, --help            show this help message and exit
  --path PATH, -p PATH  Path to data file (data.csv by default)
  --loss_control LOSS_CONTROL, -l LOSS_CONTROL
                        Epoch iterations will stop while gets loss_control value(1e-12 by default)
  --epochs EPOCHS, -e EPOCHS
                        Set the epochs number (1500 by default)
  --learning_rate ETA, -a ETA
                        Set the learning rate eta (0.2 by default)
  --loss_graphics, -g   Diagram with loss function depends on epochs
  --predict_data, -t    Diagram with data values and line prediction
  --animation, -c       Animation with prediction evolution while training
  --debug, -d           Print info about each stage of program
  --quality, -q         Model quality (R-square, MSE)

```

2. После успешной отработки программы обучения модели будет сформирован файл _coefs.csv_, в котором будут записаны найденные коэффициенты формулы вычисления предсказания цены автомобиля по заданному пробегу. <br>
В случае ошибки, будет выдано соответствующее сообщение. <br><br>
3. Для получения информации о параметрах программы, предсказывающей примерную цену автомобиля в зависимости от пробега, из папки *<your_dir_on_local_computer>* выполните *predict.py* с ключом _-h_:
```
$ python3 predict.py -h 
usage: predict.py [-h] [--debug] [--mileage MILEAGE]

options:
  -h, --help            show this help message and exit
  --debug, -d           Print info about each stage of program
  --mileage MILEAGE, -m MILEAGE
                        Car mileage for price prediction (non-negative int)
```
<br>
4. После успешной отработки программы предсказания, на экран будет выведена информация о примерной стоимости автомобиля с указанным пробегом.  <br>
В случае ошибки, будет выдано соответствующее сообщение. <br><br>

## Подробности

Подробнее о Проекте, примеры использования - по ссылке Wiki (https://github.com/dbadeev/ft_linear_regression/wiki).

<br>

## Автор

*loram (Дмитрий Бадеев)* 
