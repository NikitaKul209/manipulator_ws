
# Обучение с подкреплением манипулятора для движения в целевую точку

Необходимые пакеты Python:
- PyTorch;
- TensorFlow;
- Gym;
- Stable-baselines 3;
- Matplotlib.

## Создание и обучение 2d модели
Первый этап работ - разработка упрощённой математической модели манипулятора в двухмерном пространстве и её обучение алгоритмом PPO и DQN из пакета Stable-Baselines 3.Обучение заключается в подборе двух углов с целью попасть одним концом звена манипулятора в заданную координатами  Х  и  Y точку.


![Alt text](https://github.com/NikitaKul209/Manipulator-RL/blob/master/ScreenShots/graph.png)
 
Для обучения алгоритмом PPO требуется около 50к итераций. 
 
![Alt text](https://github.com/NikitaKul209/Manipulator-RL/blob/master/ScreenShots/PPO_2d_manipulator.png)
 
А для алгоритма DQN требуется уже около 600к итераций.

![Alt text](https://github.com/NikitaKul209/Manipulator-RL/blob/master/ScreenShots/DQN_2d_manipulator.png)


## Создание и обучение 3d модели
Второй этап - создание 3d модели манипулятора на ROS. Модель обладает двумя подвижными звеньями, закреплёнными на вращающемся на 360 градусов основании. Для изменения положения звеньев используются контроллеры ROS типа effort_controllers/JointPositionController.

![Alt text](https://github.com/NikitaKul209/Manipulator-RL/blob/master/ScreenShots/3d_manipulator.png)

