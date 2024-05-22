###############################
# Importing necessary libraries
###############################
import numpy as np
import matplotlib.pyplot as plt
from seaborn import lineplot
from pandas import DataFrame
from matplotlib.patches import Rectangle
import time

# import pi
from math import pi

# for the animation
from matplotlib.animation import FuncAnimation, writers
import matplotlib.animation as animation

# for the function get_obj_size(obj):
import gc
import sys


###############################
# This class is for the simulation='calc'
###############################
class Macromolecules:
    def __init__(self, concentration, mol_weight):
        # concentration
        self.concentration = concentration
        # molecular weight
        self.mol_weight = mol_weight

        ###########################
        # Avogadro constant - mol-1
        N_avogadro = 6.022 * 10 ** 23

        # mol per 1 ml
        mol_per_ml = concentration / mol_weight * 10 ** (-3)

        # N of molecules per 1 ml
        N_per_ml = mol_per_ml * N_avogadro

        # N of molecules per 10 mm
        N_per_10_mm = (N_per_ml) ** (1 / 3)

        # N of molecules per 1 mkm
        N_per_1_mkm = N_per_10_mm / 10_000
        ###########################

        # N of molecules per 1 mkm
        self.N_per_1_squared_mkm = int((N_per_1_mkm) ** 2)

        # see https://www.researchgate.net/post/How_can_I_convert_nm_to_kDa_for_globular_particles
        # find the radius in nm for 2D domain
        self.r_mol = (((1.212 * 10 ** (-3)) * mol_weight) * (3 / (4 * pi))) ** (1 / 2)
        self.r_mol = round(self.r_mol, 2)

        # mass of one molecule in kg
        self.mass_of_one_molecule = mol_weight * 1.66054 * 10 ** (-27)

        print(f'Number of macromolecules per squared mkm: {self.N_per_1_squared_mkm}')
        print(f'Radius of the macromolecule: {self.r_mol} nm')


###############################
# This is the main class
###############################
class Osmotic_Force_Simulation:
    def __init__(self, dict_for_the_class, simulation: str = 'calc', store_molecule_data=True):
        self.simulation = simulation
        self.store_molecule_data = store_molecule_data
        ##########
        # parameters of the simulation
        ##########
        self.simulation_domain_size = dict_for_the_class['sim_domain']  # Size of the simulation domain
        # RBC
        self.red_blood_cell_width = dict_for_the_class['RBC_w']  # Width of the red blood cells
        self.red_blood_cell_height = dict_for_the_class['RBC_h']  # Height of the red blood cells
        self.initial_intersection = 1-(dict_for_the_class['set_initial_intersection'])/100

        if simulation == 'calc':
            # domain area without the area of RBC
            domain_area_out_RBC = self.simulation_domain_size[0] * self.simulation_domain_size[
                1] - 2 * self.red_blood_cell_width * self.red_blood_cell_height

            # Macromolecules properties
            self.m = Macromolecules(dict_for_the_class['mol_concentration'], dict_for_the_class['mol_weight'])
            self.num_macromolecules = int(
                self.m.N_per_1_squared_mkm * domain_area_out_RBC)  # Number of macromolecules in the domain
            self.macromolecule_radius = self.m.r_mol / 1000  # Radius of the macromolecule in mkm
            print(f'Total number of macromolecules = {self.num_macromolecules}')

        elif simulation == 'custom':
            # macromolecules
            self.num_macromolecules = dict_for_the_class['N_mol']  # Number of macromolecules
            self.macromolecule_radius = dict_for_the_class['R_mol']  # Radius of the macromolecules in mkm
        else:
            raise ValueError('Parameter -simulation- should be -custom- or -calc-.')

        # time parameters
        self.simulation_time = dict_for_the_class['sim_time']
        self.time_step = dict_for_the_class['dt']  # Time step size
        self.num_steps = int(self.simulation_time / self.time_step)  # Number of simulation steps

        # Define the depletion force parameters
        self.depletion_force_strength = pi * self.macromolecule_radius ** 2 / (
                self.red_blood_cell_height * self.red_blood_cell_width) * self.time_step ** (
                                            -0.4) / 100  # Strength of the depletion force

        #######
        # also, we need to calculate the diffusion coefficient. see p.6 in https://doi.org/10.1140/epjh/e2020-10009-8
        #######
        self.T = dict_for_the_class['T']
        self.viscosity = dict_for_the_class['viscosity']
        self.D = (1.380649 * 10 ** (-23)) * dict_for_the_class['T'] / (
                6 * pi * dict_for_the_class['viscosity'] * self.macromolecule_radius * 10 ** (-6))

        ##########
        # arrays
        ##########
        # main arrays to collect positions at each step
        self.macromolecule_positions_history = []
        self.red_blood_cell_positions_history = []

        # Initialize positions of macromolecules
        self.macromolecule_positions = []
        self.red_blood_cell_positions = []

        ##########
        # additional parameters
        ##########
        # Parameters to control the matplotlib animation
        self.wider_domain_for_figure = 1.2
        self.shrink_the_size = 1 / 2

    def start(self, show_fig=False):
        initialize_simulation(self, show_fig=show_fig)

    def show(self, time_to_show=0):
        show_in_figure(self, time=time_to_show)

    def run(self):
        run_simulation(self)

    def animate(self, file_name='rbc_aggregation', fps_in_persent=100, frame_step=1):
        animate_simulation(self, file_name=file_name, fps_in_persent=fps_in_persent, frame_step=frame_step)

    def plot(self, save=False, title=False, name='RBC_distance'):
        plot_movement(self, save=save, title=title, name=name)

    def size(self):
        size_of_elements(self)

    # Deleting
    def __del__(self):
        pass


###############################
# Show the position of RBCs and macromolecules in the figure
###############################
def show_in_figure(self, time):
    if time > self.simulation_time:
        return print('The time is higher then the simulation time')

    # time -> N in the massive
    N_time = int(time / self.time_step)

    if N_time > len(self.macromolecule_positions_history):
        return print('Probably, you need to start the simulation via Created_object.run()')

    shrink = self.shrink_the_size  # control the resolution of the final video

    fig = plt.figure(figsize=(self.simulation_domain_size[0] * shrink, self.simulation_domain_size[1] * shrink))
    ax = fig.add_subplot(111)
    ax.axis('equal')

    plt.xlabel('Length, $\mu m$')
    plt.ylabel('Height, $\mu m$')

    x_lim = (-self.simulation_domain_size[0] * self.wider_domain_for_figure / 2,
             self.simulation_domain_size[0] * self.wider_domain_for_figure / 2)
    y_lim = (-self.simulation_domain_size[1] * self.wider_domain_for_figure / 2,
             self.simulation_domain_size[1] * self.wider_domain_for_figure / 2)
    ax.set(xlim=x_lim, ylim=y_lim)

    if self.store_molecule_data:
        ax.scatter(self.macromolecule_positions_history[N_time][:, 0],
                   self.macromolecule_positions_history[N_time][:, 1], c='blue', label='Macromolecules', s=2)
    ax.add_patch(Rectangle((self.red_blood_cell_positions_history[N_time][0][0] - self.red_blood_cell_width / 2,
                            self.red_blood_cell_positions_history[N_time][0][1] - self.red_blood_cell_height / 2),
                           self.red_blood_cell_width, self.red_blood_cell_height, edgecolor='darkred', facecolor='red',
                           fill=True, lw=2, alpha=1.0))
    ax.add_patch(Rectangle((self.red_blood_cell_positions_history[N_time][1][0] - self.red_blood_cell_width / 2,
                            self.red_blood_cell_positions_history[N_time][1][1] - self.red_blood_cell_height / 2),
                           self.red_blood_cell_width, self.red_blood_cell_height, edgecolor='darkred', facecolor='red',
                           fill=True, lw=2, alpha=1.0))
    ax.add_patch(Rectangle((-self.simulation_domain_size[0] / 2, -self.simulation_domain_size[1] / 2),
                           self.simulation_domain_size[0], self.simulation_domain_size[1], edgecolor='black',
                           facecolor=None, fill=False, lw=2, alpha=1))

    plt.show()
    return print(f'The plot is for the time {time} sec. which correspond to the {N_time}th element of a massive')


###############################
# Initialize the simulation
###############################
def initialize_simulation(self, show_fig):
    #########
    # clear all
    #########
    # main arrays to collect positions at each step
    self.macromolecule_positions_history = []
    self.red_blood_cell_positions_history = []

    # Initialize positions of macromolecules
    self.macromolecule_positions = []
    self.red_blood_cell_positions = []

    # Initialize positions and velocities of the red blood cells
    self.red_blood_cell_positions = np.array([[-self.red_blood_cell_width * self.initial_intersection / 2, -self.red_blood_cell_height / 2],
                                              [self.red_blood_cell_width * self.initial_intersection / 2,
                                               self.red_blood_cell_height / 2]])  # positions of two red blood cells

    # Initialize positions of macromolecules
    self.macromolecule_positions = np.zeros((self.num_macromolecules, 2))

    for i in range(self.num_macromolecules):
        while True:
            # Generate random positions for the macromolecule within the simulation domain
            x_rand_position = \
                np.random.uniform(low=-self.simulation_domain_size[0] / 2, high=self.simulation_domain_size[0] / 2,
                                  size=(1,))[0]
            y_rand_position = \
                np.random.uniform(low=-self.simulation_domain_size[1] / 2, high=self.simulation_domain_size[1] / 2,
                                  size=(1,))[0]

            macromolecule_position_for_one_molecule = np.array([x_rand_position, y_rand_position])

            # Check for intersection with red blood cells
            intersects = False
            for j in range(2):
                # Calculate the distance between the macromolecule and the red blood cell
                distance_x = abs(macromolecule_position_for_one_molecule[0] - self.red_blood_cell_positions[j][0])
                distance_y = abs(macromolecule_position_for_one_molecule[1] - self.red_blood_cell_positions[j][1])

                # Check if the macromolecule is within the rectangular region of the red blood cell
                if distance_x < self.red_blood_cell_width / 2 + self.macromolecule_radius and distance_y < self.red_blood_cell_height / 2 + self.macromolecule_radius:
                    intersects = True
                    break

            # If no intersection, assign the position to the macromolecule and break the loop
            if not intersects:
                self.macromolecule_positions[i] = macromolecule_position_for_one_molecule
                break

    print('The initial positions of macromolecules and RBCs are created')

    # Store the initial positions
    if self.store_molecule_data:
        self.macromolecule_positions_history.append(self.macromolecule_positions.copy())
    self.red_blood_cell_positions_history.append(self.red_blood_cell_positions.copy())

    # Convert lists to numpy arrays
    if self.store_molecule_data:
        self.macromolecule_positions_history = np.array(self.macromolecule_positions_history)
    self.red_blood_cell_positions_history = np.array(self.red_blood_cell_positions_history)

    self.macromolecule_positions = np.array(self.macromolecule_positions)

    if show_fig:
        self.show()


###############################
# Run the simulation
###############################
def run_simulation(self):
    time_begin = time.time()

    # bad points and collision variables
    bad_point = 0
    collisions = 0

    for step in range(self.num_steps):
        # make velocities of RBCs zero after each time step
        red_blood_cell_velocities = np.zeros((2, 2))
        # create macromolecule displacements under normal distibution
        macromolecule_displacements = np.random.normal(loc=0, scale=(10 ** (6)) * np.sqrt(2 * self.D * self.time_step),
                                                       size=(self.num_macromolecules, 2))
        # new positions of macromolecules
        new_macromolecule_positions = self.macromolecule_positions + macromolecule_displacements
        # calculate velocity of macromolecules
        velocity_macromolecules = macromolecule_displacements / self.time_step
        # check the intersection of macromolecules with walls
        macromolecule_displacements = np.array([check_intersection(x, v, self.simulation_domain_size) for x, v in
                                                zip(new_macromolecule_positions, macromolecule_displacements)])

        # Check to find out if the macromolecule overlaps with RBCs
        for i in range(self.num_macromolecules):
            for j in range(2):
                distance_x = abs(new_macromolecule_positions[i][0] - self.red_blood_cell_positions[j][0])
                distance_y = abs(new_macromolecule_positions[i][1] - self.red_blood_cell_positions[j][1])
                # if the overlap is true, then calculate the translated velocity to the RBC
                if distance_x < (self.red_blood_cell_width / 2 + self.macromolecule_radius) and distance_y < (
                        self.red_blood_cell_height / 2 + self.macromolecule_radius):
                    collisions += 1
                    # new displacement of i-th macromolecule
                    macromolecule_displacements[i] = np.negative(macromolecule_displacements[i])
                    # new position of i-th macromolecule
                    new_macromolecule_positions[i] = self.macromolecule_positions[i] + macromolecule_displacements[i]
                    # the velocity translated from i-th macromolecule to the j-th RBC
                    red_blood_cell_velocities[j] += velocity_macromolecules[i] * self.depletion_force_strength
                    # move RBC only alongside the "X" axis
                    red_blood_cell_velocities[j][1] = 0
                distance_x = abs(new_macromolecule_positions[i][0] - self.red_blood_cell_positions[j][0])
                distance_y = abs(new_macromolecule_positions[i][1] - self.red_blood_cell_positions[j][1])
                # check one more time because it could be a bad point
                if distance_x < (self.red_blood_cell_width / 2 + self.macromolecule_radius) and distance_y < (
                        self.red_blood_cell_height / 2 + self.macromolecule_radius):
                    bad_point += 1
                    macromolecule_displacements[i] = 0

        # Update positions
        self.macromolecule_positions += macromolecule_displacements
        self.red_blood_cell_positions += red_blood_cell_velocities * self.time_step

        # Store the positions at each time step
        if self.store_molecule_data:
            self.macromolecule_positions_history = np.append(self.macromolecule_positions_history,
                                                             [self.macromolecule_positions.copy()], axis=0)
        self.red_blood_cell_positions_history = np.append(self.red_blood_cell_positions_history,
                                                          [self.red_blood_cell_positions.copy()], axis=0)

    print(f'Simulation complete.\nSimulation time: {time.time() - time_begin: .1f} s.')
    print('___________________')
    print(f'Overall collisions: {collisions}')
    print(f'Overall bad points: {bad_point}')
    print('___________________')
    print(f'Collisions per sec.: {int(collisions / self.simulation_time)}')
    print(f'Bad points per sec.: {int(bad_point / self.simulation_time)}')
    print('___________________')
    print(f'Collisions per time step: {int(collisions / (self.simulation_time / self.time_step))}')
    print(f'Bad points per time step: {int(bad_point / (self.simulation_time / self.time_step))}')


# Function to make macromolecules to bounce from the wall
def check_intersection(x, v, domain):
    res = x.copy()
    vel = v.copy()
    if res[0] < -domain[0] / 2 or res[0] > domain[0] / 2:
        vel[0] = - vel[0]
    if res[1] < -domain[1] / 2 or res[1] > domain[1] / 2:
        vel[1] = - vel[1]
    return vel


###############################
# Animate simulation and save it in the file
###############################
def animate_simulation(self, file_name, fps_in_persent, frame_step):
    pos_array = np.array([self.red_blood_cell_width / 2, self.red_blood_cell_height / 2])

    shrink = self.shrink_the_size  # control the resolution of the final video
    # create figure for the animation
    fig = plt.figure(figsize=(self.simulation_domain_size[0] * shrink, self.simulation_domain_size[1] * shrink))
    ax = fig.add_subplot(111)
    # it is important to make axis equal
    ax.axis('equal')

    plt.xlabel('Length, $\mu m$')
    plt.ylabel('Height, $\mu m$')

    x_lim = (-self.simulation_domain_size[0] * self.wider_domain_for_figure / 2,
             self.simulation_domain_size[0] * self.wider_domain_for_figure / 2)
    y_lim = (-self.simulation_domain_size[1] * self.wider_domain_for_figure / 2,
             self.simulation_domain_size[1] * self.wider_domain_for_figure / 2)
    ax.set(xlim=x_lim, ylim=y_lim)

    # scatter and scatter2 are objects into which the positions of macromolecules and RBCs are translated
    scatter = ax.scatter([], [], c='blue', label='_Macromolecules', s=1)  # s - macromolecule_radius
    scatter2 = ax.scatter([], [], c='red', label='_Red Blood Cells', s=5)

    ax.add_patch(Rectangle((-self.simulation_domain_size[0] / 2, -self.simulation_domain_size[1] / 2),
                           self.simulation_domain_size[0], self.simulation_domain_size[1], edgecolor='black',
                           facecolor=None, fill=False, lw=2, alpha=1))
    # add time text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, bbox=dict(facecolor='w', boxstyle='round'))

    rectangles = []
    for position in self.red_blood_cell_positions_history[0]:
        rect = Rectangle(position - pos_array, self.red_blood_cell_width, self.red_blood_cell_height,
                         edgecolor='darkred', facecolor='red', fill=True, lw=2, alpha=1.0)
        rectangles.append(rect)
        ax.add_patch(rect)

    def update(frame):
        if self.store_molecule_data:
            scatter.set_offsets(self.macromolecule_positions_history[frame * frame_step])
        scatter2.set_offsets(self.red_blood_cell_positions_history[frame * frame_step])
        time_text.set_text('Time Step: {} s.'.format(round(frame * frame_step * self.time_step, 2)))
        # Update the positions of the rectangles
        for rect, position in zip(rectangles, self.red_blood_cell_positions_history[frame * frame_step]):
            rect.set_xy(position - pos_array)

        return scatter, scatter2, *rectangles

    # create animation using FuncAnimation
    ani = FuncAnimation(fig, update, frames=int(len(self.red_blood_cell_positions_history) / frame_step),
                        interval=self.time_step * 1000, blit=True)
    # show the animation
    plt.show()

    # save animation
    FFwriter = animation.FFMpegWriter(fps=int((fps_in_persent / 100) * 1 / (frame_step * self.time_step)),
                                      bitrate=20000)
    ani.save(file_name + '.mp4', writer=FFwriter, dpi=200)

    print(f'Animation was saved in {file_name}' + '.mp4')


###############################
# Plot the movement of the RBC centers
###############################
def plot_movement(self, save, title, name):
    # calculate distance between RBC centers
    def distance(positions, frame):
        return -(positions[frame][0][0] - positions[frame][1][0])

    distance_RBC_agg = [distance(self.red_blood_cell_positions_history, j) for j in
                        range(len(self.red_blood_cell_positions_history))]
    intersection_RBC_agg = [100 * (1 - x / self.red_blood_cell_width) for x in distance_RBC_agg]

    time_overall = [j * self.time_step for j in range(len(self.red_blood_cell_positions_history))]

    main_data = DataFrame({'Time, s.': time_overall, 'Distance between\nRBC centers, a.u.': distance_RBC_agg,
                           'Intersection of RBCs, %': intersection_RBC_agg})
    # create two figures
    fig, ax = plt.subplots(2, constrained_layout=True)

    if title and self.simulation == 'calc':
        fig.suptitle(
            f'Concentration = {self.m.concentration} $mg/ml$. Molecular weight = {self.m.mol_weight:,} $Da$.\nTemperature = {self.T} $K$. Viscosity = {self.viscosity} $Pa*s$.')
    elif title and self.simulation == 'custom':
        fig.suptitle(
            f'Macromolecule radius = {self.macromolecule_radius} $\mu m$. Number of macromolecules = {self.num_macromolecules:,}.\nTemperature = {self.T} $K$. Viscosity = {self.viscosity} $Pa*s$.')

    lineplot(x='Time, s.', y='Distance between\nRBC centers, a.u.', data=main_data, ax=ax[0])
    lineplot(x='Time, s.', y='Intersection of RBCs, %', data=main_data, ax=ax[1])
    plt.show()
    # save figure if necessary
    if save:
        fig.savefig(name + '.png', dpi=600)
        main_data.to_excel(name + '.xlsx', sheet_name='RBC_aggregation')


###############################
# Get the size of the lists
###############################
def size_of_elements(self):
    print(
        f'The total size for the macromolecular positions: {get_obj_size(self.macromolecule_positions_history) / (10 ** 6)} MB')
    print(f'The total size for the RBC positions: {get_obj_size(self.red_blood_cell_positions_history) / (10 ** 6)} MB')


# see https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python
def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0
    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))
        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))
        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}
        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())
    return sz
