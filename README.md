# Robôs Autônomos - trabalho 2

Cálculo da área de um ambiente desconhecido utilizando o robô móvel Clearpath Jackal e laser 2D SICK LMS511.

### Requisitos

* Python (>=2.7, >=3.6)
* ROS (apenas ROS Melodic foi testado)          | `$ source install/intall_ros_melodic.sh`
* Xterm; Gazebo; Clearpath Jackal simulator     | `$ source install/install_ros_packages.sh`  (versões para melodic)
* Numpy; Scipy; Matplotlib                      | `$ pip install -r requirements.txt`

### Build & Run

* Compilar pacote 'slam_area'                   | `$ source build.sh`
* Rodar launch de simulação                     | `$ source launch.sh <world1 ou world2>`

### Solução proposta

* Memorizar pontos obtidos do sensor para mapear o ambiente, usando o robô como referência.
* Estimar área pela projeção das paredes com o mapa de pontos.
* Robô móvel como seguidor de parede, deve funcionar apenas em ambientes fechados.
* Robô pode ser posto em qualquer posição do ambiente.
* Controle e mapeamento são funções separadas e independentes.

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
* Computa a área do polígono gerado (fórmula shoelace).

### Problemas existentes

* Controle do robô instável, não suavizado.
* Drift de odometria.
* Pontos do sensor atribuídos em locais errados.
* Detecção de parada com problemas se pontos de referência não são coerentes.
* Valor de área encontrada incorreto se pontos mal localizados.

### Possíveis melhorias

* EKF Slam (Extended Kalman FIlter) para correção da localização.
* Detecção de segmentos de reta e filtragem de outliers (Split and Merge).
* Melhor detecção de fechamento da área pelos pontos analisados.
* Melhorar controle do robô móvel.
* Fazer SLAM online, utilizando os dados obtidos do sensor.
* Pode ser viabilizado para ambientes fechados após melhorias, como túneis.
* Possibilitar o uso em ambientes externos, outra abordagem de controle, não seguidor de parede.

### Referências

* UFES - Disciplina de Tópicos em Robôs Autônomos (2022)
* Augusto Abling