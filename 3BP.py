import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import time

r_scale = 0.9
v_scale = 0.45
pos = np.random.uniform(-r_scale, r_scale, (3,3))
vel = np.random.normal(0, v_scale, (3,3))

com_pos = np.mean(pos, axis=0)
pos -= com_pos

com_vel = np.mean(vel, axis=0)
vel -= com_vel

# mass
m1 = 1.0
m2 = 1.0
m3 = 1.0

# Pos
initial_position1 = pos[0].tolist()
initial_position2 = pos[1].tolist()
initial_position3 = pos[2].tolist()

# vel
initial_velocity1 = vel[0].tolist()
initial_velocity2 = vel[1].tolist()
initial_velocity3 = vel[2].tolist()

initial_conditions = np.array([
    initial_position1, initial_position2, initial_position3,
    initial_velocity1, initial_velocity2, initial_velocity3
]).ravel()

def ODEs(t, s, m1, m2, m3):
    r1, r2, r3 = s[0:3], s[3:6], s[6:9]
    dp1_dt, dp2_dt, dp3_dt = s[9:12], s[12:15], s[15:18]

    p1, p2, p3 = dp1_dt, dp2_dt, dp3_dt

    dp1_dt = m2 * (r2 - r1) / np.linalg.norm(r2 - r1) ** 3 + m3 * (r3 - r1) / np.linalg.norm(r3 - r1) ** 3
    dp2_dt = m3 * (r3 - r2) / np.linalg.norm(r3 - r2) ** 3 + m1 * (r1 - r2) / np.linalg.norm(r1 - r2) ** 3
    dp3_dt = m2 * (r2 - r3) / np.linalg.norm(r2 - r3) ** 3 + m1 * (r1 - r3) / np.linalg.norm(r1 - r3) ** 3

    return np.array([p1, p2, p3, dp1_dt, dp2_dt, dp3_dt]).ravel()

time_s, time_e = 0, 7
t_points = np.linspace(time_s, time_e, 1001)

t1=time.time()
solution = solve_ivp(
    fun=ODEs,
    t_span=(time_s, time_e),
    y0=initial_conditions,
    t_eval=t_points,
    args=(m1, m2, m3)
)
t2=time.time()
print(f"Solved in: {t2-t1:.3f} [s]")

t_sol = solution.t
p1x_sol = solution.y[0]
p1y_sol = solution.y[1]
p1z_sol = solution.y[2]

p2x_sol = solution.y[3]
p2y_sol = solution.y[4]
p2z_sol = solution.y[5]

p3x_sol = solution.y[6]
p3y_sol = solution.y[7]
p3z_sol = solution.y[8]

fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

fig.patch.set_facecolor('black')
ax.set_facecolor('black')

ax.title.set_color('white')
ax.set_xlabel("x", color='white')
ax.set_ylabel("y", color='white')
ax.set_zlabel("z", color='white')

ax.tick_params(colors='white', which='both')

star1_plot, = ax.plot(p1x_sol, p1y_sol, p1z_sol, 'yellow', label='Star 1', linewidth=1)
star2_plot, = ax.plot(p2x_sol, p2y_sol, p2z_sol, 'red', label='Star 2', linewidth=1)
star3_plot, = ax.plot(p3x_sol, p3y_sol, p3z_sol, 'blue', label='Star 3', linewidth=1)

star1_dot, = ax.plot([p1x_sol[-1]], [p1y_sol[-1]], [p1z_sol[-1]], 'o', color='yellow', markersize=6)
star2_dot, = ax.plot([p2x_sol[-1]], [p2y_sol[-1]], [p2z_sol[-1]], 'o', color='red', markersize=6)
star3_dot, = ax.plot([p3x_sol[-1]], [p3y_sol[-1]], [p3z_sol[-1]], 'o', color='blue', markersize=6)

ax.set_title("The 3 Body Problem")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.grid()
plt.legend()

def update(frame):
    lower_lim = max(0, frame - 300)
    print(f"Progress: {(frame+1)/len(t_points):.1%} | 100.0 %", end='\r')

    x_current_1 = p1x_sol[lower_lim:frame+1]
    y_current_1 = p1y_sol[lower_lim:frame+1]
    z_current_1 = p1z_sol[lower_lim:frame+1]

    x_current_2 = p2x_sol[lower_lim:frame+1]
    y_current_2 = p2y_sol[lower_lim:frame+1]
    z_current_2 = p2z_sol[lower_lim:frame+1]

    x_current_3 = p3x_sol[lower_lim:frame+1]
    y_current_3 = p3y_sol[lower_lim:frame+1]
    z_current_3 = p3z_sol[lower_lim:frame+1]

    star1_plot.set_data(x_current_1, y_current_1)
    star1_plot.set_3d_properties(z_current_1)

    star1_dot.set_data([x_current_1[-1]], [y_current_1[-1]])
    star1_dot.set_3d_properties([z_current_1[-1]])



    star2_plot.set_data(x_current_2, y_current_2)
    star2_plot.set_3d_properties(z_current_2)

    star2_dot.set_data([x_current_2[-1]], [y_current_2[-1]])
    star2_dot.set_3d_properties([z_current_2[-1]])



    star3_plot.set_data(x_current_3, y_current_3)
    star3_plot.set_3d_properties(z_current_3)

    star3_dot.set_data([x_current_3[-1]], [y_current_3[-1]])
    star3_dot.set_3d_properties([z_current_3[-1]])


    return star1_plot, star1_dot, star2_plot, star2_dot, star3_plot, star3_dot

animation = FuncAnimation(fig, update, frames=range(0, len(t_points), 2), interval=10, blit=True)
plt.show()