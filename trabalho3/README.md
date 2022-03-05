# Robôs Autônomos - trabalho 3

EKF SLAM aplicado a um robô autônomo para correção de localização e mapeamento do ambiente.

### Requisitos

* Python (>=2.7, >=3.6)
* ROS (apenas ROS Melodic foi testado) | `$ source install/intall_ros_melodic.sh`
* Gazebo; Clearpath Jackal simulator | `$ source install/install_ros_packages.sh`  (versões para melodic)
* Numpy; Scipy; Matplotlib; Shapely | `$ pip install -r requirements.txt`

### Build & Run

* Compilar pacote 'slam_ekf' | `$ source build.sh`
* Rodar launch de simulação | `$ source launch.sh <world>`

### Solução proposta

Esta proposta de projeto apresenta as motivações e planejamento para resolução de problemas relacionados ao mapeamento, já conhecidos na literatura. Através dos erros gerados no referencial de odometria de um robô móvel, as características de um ambiente interno desconhecido são mapeadas incorretamente. A abordagem EKF SLAM foi selecionada por estar consolidada no meio acadêmico e prover possíveis bons resultados, onde a localização é corrigida pela predição de movimento baseada em filtro de Kalman. O experimento será realizado em simulação virtual, com a representação de uma sala de escritório, onde o robô móvel, com uma trajetória determinada, terá que mapear o ambiente de forma coerente e apresentar baixo desvio de localização dos pontos que formam o mapa real.

### Planejamento

* Criar pacote para EKF SLAM | `catkin_create_pkg slam_ekf std_msgs rospy roscpp`
* Selecionar/criar mundo para simulações | Escolhido foi...
* Editar trajetória | Editar arquivos em *config/* *goals_from_csv.yaml* e *path_points.csv* de path_generator
* Desenvolver script para mapeamento corrigido com EKF | *scripts/ekf_mapping.py*
* Comparar resultados com o mundo real simulado | Métrica por IOU, alguma outra, ou apenas visualmente com o mapa real?

### Referências

* UFES - Disciplina de Tópicos em Robôs Autônomos (2022)
* Augusto Abling