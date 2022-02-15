# Robôs Autônomos - trabalho 2

Cálculo da área de um ambiente desconhecido utilizando robô móvel e sensoriamento.

### Controle do robô

* Robô tenta se localizar pela detecção de 5 amostras do laser.
* Lateral esquerda, lateral frontal esquerda, frontal, lateral frontal direita e lateral direita.
* Controla apenas a velocidade linear em “x” e velocidade angular em “z”.
* Começa buscando parede mais próxima para iniciar o seguimento.
* Tenta manter direção no sentido horário, muda quando detecta parede ou obstáculo.
* Robô para apenas se receber retorno do mapeamento indicando finalização, se não matem-se em movimento.

### Mapeamento do ambiente

* Área computada ao final do mapeamento.
* Salva histórico de posições pela odometria.
* Salva pontos escaneados pelo laser (landmarks), quantidade pelos parâmetros.
* Atribuição de novos pontos conforme distância euclidiana, limites pelos parâmetros.
* Salva pontos de checagem (checkpoints) das posições do robô.
* Detecta final do mapeamento pela proximidade dos checkpoints.
* Detecta final do mapeamento pela proximidade dos landmarks ordenados.
* Ao finalizar gera polígono que representa a área externa dos landmarks (Convex Hull).
* Computa a área do polígono gerado (fórmula de Shoelace).


## Requisitos

* Python (>=2.7, >=3.6)
* ROS (apenas ROS Melodic foi testado)          | `$ source install/intall_ros_melodic.sh`
* Xterm; Gazebo; Clearpath Jackal simulator     | `$ source install/install_ros_packages.sh`

# Build & Run

* Compilar pacote 'slam_area'                   | `$ source build.sh`
* Rodar launch de simulação                     | `$ source launch.sh`