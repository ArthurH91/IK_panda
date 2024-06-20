from os.path import join, dirname, abspath

import pinocchio as pin
import numpy as np
from scipy.optimize import minimize

# from wrapper_panda import PandaWrapper
from wrapper_meshcat import MeshcatWrapper

# Creating the robot

urdf_model_path = "/home/arthur/Desktop/Code/test/forkrzchan/asdf/file.urdf"
mesh_dir = "/home/arthur/Desktop/Code/test/forkrzchan/asdf/meshes"
urdf_filename = "file.urdf"
(
    rmodel,
    cmodel,
    vmodel,
) = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir)

q0 = pin.neutral(rmodel)

# Locking the gripper
jointsToLockIDs = [8, 9]

geom_models = [vmodel, cmodel]
model_reduced, geometric_models_reduced = pin.buildReducedModel(
    rmodel,
    list_of_geom_models=geom_models,
    list_of_joints_to_lock=jointsToLockIDs,
    reference_configuration=q0,
)

vmodel_reduced, cmodel_reduced = (
    geometric_models_reduced[0],
    geometric_models_reduced[1],
)

# robot_wrapper = PandaWrapper(capsule=True, auto_col=True)
# rmodel, cmodel, vmodel = robot_wrapper()
# rdata = rmodel.createData()
# cdata = cmodel.createData()
# # Generating the meshcat visualizer
MeshcatVis = MeshcatWrapper()
vis = MeshcatVis.visualize(
    robot_model=model_reduced, robot_visual_model=cmodel_reduced, robot_collision_model=cmodel_reduced
)

rdata = model_reduced.createData()

# Desired end-effector pose
desired_position = np.array( [0.122, -0.217, 1.50])
desired_orientation = pin.utils.rpyToMatrix(0.0,np.pi,np.pi/2)
desired_transform = pin.SE3(desired_orientation, desired_position)

# Initial joint configuration
q_initial = pin.neutral(model_reduced)

q_initial = np.array([0.003777164052342363, -0.8137102209141379, 0.014652104986350651, -2.343809671909919, 0.007626496720331425, 1.769832534286711, 0.7945119172336511])
vis[0].display(q_initial)
input()
# Objective function
def objective_function(q):
    pin.forwardKinematics(model_reduced, rdata, q)
    pin.updateFramePlacements(model_reduced, rdata)
    M = rdata.oMf[
        model_reduced.getFrameId("camera_color_optical_frame")
    ]  # Assuming the end-effector is the last frame
    error = pin.log6(M.inverse() * desired_transform).vector
    q_reg = q-q_initial
    return np.sum(error**2) + np.sum(q_reg**2)
    # return np.sum(error**2)
# Joint limits constraints
def joint_limits_constraints(q):
    lower_limits = model_reduced.lowerPositionLimit
    upper_limits = model_reduced.upperPositionLimit
    return np.concatenate((q - lower_limits, upper_limits - q))

constraints = {'type': 'ineq', 'fun': joint_limits_constraints}
    

# Optimize
result = minimize(objective_function, q_initial, constraints=constraints, method="BFGS", options={"disp": True})

# Check result
if result.success:
    q_solution = result.x
    print("Found solution:", q_solution)
else:
    print("Optimization failed.")

vis[0].display(q_solution)

print(q_solution - np.array([np.pi]*7))