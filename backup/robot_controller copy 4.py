#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import rospy
import numpy as np


# Odometria contendo posição, orientação, vel. linear e angular
from nav_msgs.msg import Odometry

# Twist é a velocidade linear e angular
from geometry_msgs.msg import Twist

# Amostras de distâncias obtidas pelo LaserScan
from sensor_msgs.msg import LaserScan

# Mapping
from ekf_slam.robot import Robot
from ekf_slam.plotmap import plotMap, plotEstimate, plotMeasurement, plotError
from ekf_slam.ekf import predict, update


class Controller():
    def __init__(self):
        self.odom_subscriber = rospy.Subscriber(
            name='/jackal_velocity_controller/odom',
            data_class=Odometry,
            callback=self.odom_callback,
            queue_size=10
        )

        self.velocity_subscriber = rospy.Subscriber(
            name='/jackal_velocity_controller/cmd_vel',
            data_class=Twist,
            callback=self.velocity_callback,
            queue_size=10
        )

        self.scan_subscription = rospy.Subscriber(
            name='/front/scan',
            data_class=LaserScan,
            callback=self.scan_callback,
            queue_size=10
        )

        self.velocity_publisher = rospy.Publisher(
            name='/jackal_velocity_controller/cmd_vel',
            data_class=Twist,
            queue_size=10)

        # Posições de amostragem para LaserScan e distâncias iniciais
        self.left_dist = 99999.
        self.leftfront_dist = 99999.
        self.front_dist = 99999.
        self.rightfront_dist = 99999.
        self.right_dist = 99999.

        # Posição do robô
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_yaw = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Velocidades
        self.speed_linear_fast = 0.20  # m/s
        self.speed_linear_slow = 0.1  # m/s
        self.speed_angular_fast = 2.0  # rad/s
        self.speed_angular_slow = 0.3  # rad/s

        # Estado do seguidor de parede
        self.robot_state = "turn left"

        # Distância a ser mantida da parede
        self.dist_wall_close = 0.5  # metros
        self.dist_wall_thresh = 0.7  # metros

        self.search_wall = True
        self.track_wall = False
        self.finish = False

    # Converte quaternions para ângulos de Euler
    def euler_from_quaternion(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # radianos

    # Callback de odometria do robô (posição e orientação)
    # em '/jackal_velocity_controller/odom'
    def odom_callback(self, msg):
        # Posições x, y, z.
        # Orientações x, y, z, w quaternion
        roll, pitch, yaw = self.euler_from_quaternion(
          msg.pose.pose.orientation.x,
          msg.pose.pose.orientation.y,
          msg.pose.pose.orientation.z,
          msg.pose.pose.orientation.w)

        obs_state_vector_x_y_yaw = [
            msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]

        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = yaw

    # Callback das velocidades lineares e angulares
    # em '/jackal_velocity_controller/cmd_vel'
    def velocity_callback(self, msg):
        # Velocidade linear em frente para o robô
        v = msg.linear.x

        # Velicidade angular em torno do eixo z do robô
        yaw_rate = msg.angular.z

    # Callback do LaserScan
    # em '/front/laser'
    def scan_callback(self, msg):
        # n_laser_beams = len(msg.ranges) = 720
        # max_angle = 270
        # r = n_laser_beams/max_angle = 2.6666 -> 2.6666*angle
        self.left_dist = msg.ranges[600]  # 225
        self.leftfront_dist = msg.ranges[480]  # 180
        self.front_dist = msg.ranges[360]  # 135
        self.rightfront_dist = msg.ranges[240]  # 90
        self.right_dist = msg.ranges[120]  # 45

    # Segue parede
    def follow_wall(self):
        # Cria mensagem Twist
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        # Distância da parede
        d1 = self.dist_wall_close
        d2 = self.dist_wall_thresh

        if self.search_wall:
            self.robot_state = "searching wall..."
            if self.front_dist > d2 and self.rightfront_dist > d2:
                msg.linear.x = self.speed_linear_fast
                if self.rightfront_dist < self.front_dist:
                    msg.angular.z = -self.speed_angular_slow
                else:
                    msg.angular.z = 0
            else:
                self.search_wall = False
                self.start_x = self.current_x
                self.start_y = self.current_y
                self.start_yaw = self.current_yaw

        elif (
                abs(self.current_x - self.start_x) > 15 and
                abs(self.current_y - self.start_y) > 15 and
                abs(self.current_yaw - self.start_yaw) > 15):
            self.track_wall = True

        elif self.track_wall and (
                abs(self.current_x - self.start_x) < 10 and
                abs(self.current_y - self.start_y) < 10 and
                abs(self.current_yaw - self.start_yaw) < 10):
            msg.linear.x = 0
            msg.angular.z = 0
            self.robot_state = "finish"
        else:
            if self.leftfront_dist < d1 or self.front_dist < d1 or\
                    self.rightfront_dist < d1:
                self.robot_state = "stop"
                msg.linear.x = 0
            elif self.leftfront_dist < d2 or self.front_dist < d2 or\
                    self.rightfront_dist < d2:
                self.robot_state = "slow"
                msg.linear.x = self.speed_linear_slow
            else:
                self.robot_state = "fast"
                msg.linear.x = self.speed_linear_fast

            if self.front_dist < d2 or self.rightfront_dist < d1:
                self.robot_state += " turning fast inv."
                msg.angular.z = self.speed_angular_fast
            elif self.right_dist < d1 or self.rightfront_dist < d2:
                self.robot_state += " turning slow inv."
                msg.angular.z = self.speed_angular_slow
            elif self.front_dist > d1+d2 and self.rightfront_dist > d1+d2 and\
                    self.right_dist > d1+d2:
                self.robot_state += " turning fast to wall"
                msg.angular.z = -self.speed_angular_fast
            else:
                self.robot_state += " turning slow to wall"
                msg.angular.z = -self.speed_angular_slow

        # Envia mensagem da velocidade atualizada
        self.velocity_publisher.publish(msg)

        # Mostra distâncias detectadas pelo LaserScan
        rospy.loginfo(
            " 180=" + str(round(self.left_dist, 2)) +
            " 135=" + str(round(self.leftfront_dist, 2)) +
            " 90=" + str(round(self.front_dist, 2)) +
            " 45=" + str(round(self.rightfront_dist, 2)) +
            " 0=" + str(round(self.right_dist, 2)) +
            " " + self.robot_state
        )


class EKFMapping():
    def init(self):
        n = 50  # number of static landmarks
        self.mapsize = 40
        landmark_xy = self.mapsize*(np.random.rand(n, 2) - 0.5)
        landmark_id = np.transpose([np.linspace(0, n-1, n, dtype='uint16')])
        ls = np.append(landmark_xy, landmark_id, axis=1)

        # In[Generate dynamic landmarks]
        k = 0  # number of dynamic landmarks
        vm = 5  # velocity multiplier
        landmark_xy = self.mapsize*(np.random.rand(k, 2) - 0.5)
        landmark_v = np.random.rand(k, 2) - 0.5
        landmark_id = np.transpose([np.linspace(n, n+k-1, k, dtype='uint16')])
        ld = np.append(landmark_xy, landmark_id, axis=1)
        ld = np.append(ld, landmark_v, axis=1)

        # In[Define and initialize robot parameters]
        fov = 120
        self.Rt = 5*np.array([
            [0.1, 0, 0],
            [0, 0.01, 0],
            [0, 0, 0.01]])
        Qt = np.array([
            [0.01, 0],
            [0, 0.01]])

        x_init = [0, 0, 0.5*np.pi]

        self.r1 = Robot(x_init, fov, self.Rt, Qt)

        # In[Generate inputs and measurements]

        steps = 30
        stepsize = 3
        curviness = 0.5

        x_true = [x_init]
        obs = []

        # generate input sequence
        u = np.zeros((steps, 3))
        u[:, 0] = stepsize
        u[4:12, 1] = curviness
        u[18:26, 1] = curviness

        # Generate random trajectory instead
        # u = np.append(stepsize*np.ones((steps,1),dtype='uint8'),
        #              curviness*np.random.randn(steps,2),
        #              axis=1)

        # generate dynamic landmark trajectories
        ldt = ld
        for j in range(1, steps):
            # update dynamic landmarks
            F = np.array([
                [1, 0, 0, vm, 0],
                [0, 1, 0, 0, vm],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]])
            for i in range(len(ld)):
                ld[i, :] = F.dot(ld[i, :].T).T
            ldt = np.dstack((ldt, ld))

        # generate robot states and observations
        for movement, t in zip(u, range(steps)):
            landmarks = np.append(ls, ldt[:, :3, t], axis=0)

            # process robot movement
            x_true.append(self.r1.move(movement))
            obs.append(self.r1.sense(landmarks))

        plotMap(ls, ldt, x_true, self.r1, self.mapsize)

        # In[Estimation]

        # Initialize state matrices
        inf = 1e6

        self.mu = np.append(np.array([x_init]).T, np.zeros((2*(n+k), 1)), axis=0)
        self.mu_new = self.mu

        self.cov = inf*np.eye(2*(n+k)+3)
        self.cov[:3, :3] = np.zeros((3, 3))

        c_prob = 0.5*np.ones((n+k, 1))

        plotEstimate(self.mu, self.cov, self.r1, self.mapsize)

    def odometry_model(self, pose, odometry):
        rx, ry, rθ = pose
        direction = rθ + odometry.rot1
        rx += odometry.trans * math.cos(direction)
        ry += odometry.trans * math.sin(direction)
        rθ += odometry.rot1 + odometry.rot2
        rθ = rem2pi(rθ, RoundNearest)  # Round to [-π, π]
        return [rx, ry, rθ]

    def observation_model(self, robot_pose, range_bearing):
        rx, ry, rθ = robot_pose
        range, bearing = range_bearing.range, range_bearing.bearing
        mx = rx + range * math.cos(bearing + rθ)
        my = ry + range * math.sin(bearing + rθ)
        return [mx, my]

    def belief_init(num_landmarks):
        μ = Vector{Union{Float32, Missing}}(missing, 3 + 2*num_landmarks)
        μ[1:3] .= 0
        Σ = zeros(Float32, 3+2*num_landmarks, 3+2*num_landmarks)
        Σ[diagind(Σ)[1:3]] .= 0
        Σ[diagind(Σ)[4:end]] .= 1000

        return Belief(μ, Symmetric(Σ))

function prediction_step(belief, odometry)
    # Compute the new mu based on the noise-free (odometry-based) motion model
    rx, ry, rθ = belief.mean[1:3]
    belief.mean[1:3] = standard_odometry_model([rx, ry, rθ], odometry)

    # Compute the 3x3 Jacobian Gx of the motion model
    Gx = Matrix{Float32}(I, 3, 3)
    heading = rθ + odometry.rot1
    Gx[1, 3] -= odometry.trans * sin(heading)  # ∂x'/∂θ
    Gx[2, 3] += odometry.trans * cos(heading)  # ∂y'/∂θ

    # Motion noise
    Rx = Diagonal{Float32}([0.1, 0.1, 0.01])

    # Compute the predicted sigma after incorporating the motion
    Σxx = belief.covariance[1:3, 1:3]
    Σxm = belief.covariance[1:3, 4:end]

    Σ = Matrix(belief.covariance)
	Σ[1:3, 1:3] = Gx * Σxx * Gx' + Rx
    Σ[1:3, 4:end] = Gx * Σxm
	belief.covariance = Symmetric(Σ)

end


function correction_step(belief, range_bearings)
	rx, ry, rθ = belief.mean[1:3]

	num_range_bearings = length(range_bearings)
	num_dim_state = length(belief.mean)

	H = Matrix{Float32}(undef, 2 * num_range_bearings, num_dim_state) # Jacobian matrix ∂ẑ/∂(rx,ry)
	zs, ẑs = [], []  # true and predicted observations

    for (i, range_bearing) in enumerate(range_bearings)
		mid = range_bearing.landmark_id
        if ismissing(belief.mean[2*mid+2])
			# Initialize its pose in mu based on the measurement and the current robot pose
			mx, my = range_bearing_model([rx, ry, rθ], range_bearing)
			belief.mean[2*mid+2:2*mid+3] = [mx, my]
        end
		# Add the landmark measurement to the Z vector
		zs = [zs; range_bearing.range; range_bearing.bearing]

		# Use the current estimate of the landmark pose
		# to compute the corresponding expected measurement in z̄:
		mx, my = belief.mean[2*mid+2:2*mid+3]
		δ = [mx - rx, my - ry]
		q = dot(δ, δ)
		sqrtq = sqrt(q)

	 	ẑs = [ẑs; sqrtq; atan(δ[2], δ[1]) - rθ]

		# Compute the Jacobian Hi of the measurement function h for this observation
		δx, δy = δ
		Hi = zeros(Float32, 2, num_dim_state)
		Hi[1:2, 1:3] = [
			-sqrtq * δx  -sqrtq * δy   0;
			δy           -δx           -q
		] / q
		Hi[1:2, 2*mid+2:2*mid+3] = [
			sqrtq * δx sqrtq * δy;
			-δy δx
		] / q

		# Augment H with the new Hi
		H[2*i-1:2*i, 1:end] = Hi
    end

	# Construct the sensor noise matrix Q
	Q = Diagonal{Float32}(ones(2 * num_range_bearings) * 0.01)

	# Compute the Kalman gain K
	K = belief.covariance * H' * inv(H * belief.covariance * H' + Q)

	# Compute the difference between the expected and recorded measurements.
	Δz = zs - ẑs
	# Normalize the bearings
	Δz[2:2:end] = map(bearing->rem2pi(bearing, RoundNearest), Δz[2:2:end])

	# Finish the correction step by computing the new mu and sigma.
	belief.mean += K * Δz
	I = Diagonal{Float32}(ones(num_dim_state))
	belief.covariance = Symmetric((I - K * H) * belief.covariance)

	# Normalize theta in the robot pose.
	belief.mean[3] = rem2pi(belief.mean[3], RoundNearest)
end

    def update_map(self, movement, measurement):
        self.mu_new, self.cov = predict(self.mu_new, self.cov, movement, self.Rt)
        self.mu = np.append(self.mu, self.mu_new, axis=1)
        plotEstimate(self.mu, self.cov, self.r1, self.mapsize)

        # print('Measurements: {0:d}'.format(len(measurement)))
        # mu_new, cov, c_prob_new = update(mu_new, cov, measurement, c_prob[:,-1].reshape(n+k,1), Qt)
        # mu = np.append(mu,mu_new,axis=1)
        # c_prob = np.append(c_prob, c_prob_new, axis=1)
        # plotEstimate(mu, cov, self.r1, self.mapsize)
        # plotMeasurement(mu_new, cov, measurement, n)

        # plotError(mu,x_true[:len(mu[:,0::2])][:])
        print('----------')


def main():
    # Cria node do controlador do robô
    rospy.init_node('robosaut_controller', anonymous=True)
    controller = Controller()
    mapping = EKFMapping()

    rate = rospy.Rate(10)  # 10hz

    # Espera tópico do laser abrir
    data = None
    while data is None:
        try:
            data = rospy.wait_for_message('/front/scan', LaserScan, timeout=2)
        except Exception:
            pass

    while not rospy.is_shutdown() or controller.finish:
        controller.follow_wall()
        mapping.update_map(
            movement=[
                controller.current_x,
                controller.current_y,
                controller.current_yaw
            ],
            measurement=[]
        )
        rate.sleep()

    # Aguarda finalizar o processo
    rospy.spin()
    del controller


if __name__ == "__main__":
    main()
