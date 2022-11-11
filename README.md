
# Обучение с подкреплением манипулятора для движения в целевую точку

Необходимые пакеты Python:
- PyTorch;
- TensorFlow;
- Gym;
- Stable-baselines 3;
- Matplotlib.

## Создание и обучение 2d модели
Разработка упрощённой математической модели манипулятора в двухмерном пространстве и её обучение алгоритмом PPO и DQN из пакета Stable-Baselines 3.Обучение заключается в подборе двух углов с целью попасть одним концом звена манипулятора в заданную координатами  Х  и  Y точку.


![Alt text](https://github.com/NikitaKul209/Manipulator-RL/blob/master/ScreenShots/graph.png)
 
Для обучения алгоритмом PPO требуется около 50к итераций. 
 
![Alt text](https://github.com/NikitaKul209/Manipulator-RL/blob/master/ScreenShots/PPO_2d_manipulator.png)
 
А для алгоритма DQN требуется уже около 600к итераций.

![Alt text](https://github.com/NikitaKul209/Manipulator-RL/blob/master/ScreenShots/DQN_2d_manipulator.png)


Для запуска обучения вызвать в скрипте 2d_manipulator.py функцию "train(model_name,num_timesteps,algorithm)",где 
- "model_name" - имя сохранённой модели после обучения;
- "num_timesteps" - количество итераций в обучении;
- "algorithm" - название алгоритма. PPO или DQN.
Перед этим необходимо задать координаты для обуения,вызвав из класса "env" переменные "x_goal" и "y_goal".
Для запуска тестирования обученной модели вызвать в скрипте 2d_manipulator.py функцию "test(model_name,algorithm)"
Для запуска дообучения уже обученной модели вызвать в скрипте 2d_manipulator.py функцию "train_old_model(model_name,num_timesteps,algorithm)"

## Создание и обучение 3d модели
Создание 3d модели манипулятора на ROS. Модель обладает двумя подвижными звеньями, закреплёнными на вращающемся на 360 градусов основании. Для изменения положения звеньев используются контроллеры ROS типа effort_controllers/JointPositionController.

![Alt text](https://github.com/NikitaKul209/Manipulator-RL/blob/master/ScreenShots/3d_manipulator.png)

Для обучения алгоритмом PPO требуется около 70к итераций. 

![Alt text](https://github.com/NikitaKul209/Manipulator-RL/blob/master/ScreenShots/PPO_3d_manipulator.png)

Для запуска обучения необходимо:
- перейти в дирректорию пакета ROS - "cd manipulator_ws";
- "source devel/setup.bash";
- запустить Gazebo с моделью манипулятора - "roslaunch my_robot_description launch.launch";
- запустить обучающий скрипт - "rosrun my_robot_description main_robot_control.py". 

Для обучения в скрипте main_robot_control.py должна быть вызвана функция "train(model_name,algorithm,num_timesteps)"
Для запуска тестирования обученной модели вызвать в скрипте main_robot_control.py функцию "test(model_name,algorithm)"
Для запуска дообучения уже обученной модели вызвать в скрипте main_robot_control.py функцию "train_old_model(algorithm,model_name,num_timesteps))"
