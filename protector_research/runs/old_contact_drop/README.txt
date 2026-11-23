Drop tests with the OLD (bench) simulator's unprotected contact, to identify the ball/ground stiffness it implies.

Contact exactly as in the bench robot model (pm_v2_mesh.xml, mujoco 3.2.3): the ball geom carries the robot collision
class (solref 0.0005 1, solimp 0.9 0.95 [0.001 0.5 2], condim 6), the floor is a plane with MuJoCo defaults
(solref 0.02 1); equal solmix/priority -> mixed contact solref 0.01025. Options as serial_pm_v2_mesh.xml:
timestep 2 ms, tolerance 1e-10, Euler, pyramidal, Newton (defaults). No CHR scaling (protection off).
Aluminum balls (rho 2700) of several masses from several heights; contact starts at 4 phases of a step.

Writes runs/old_contact_drop/: one CSV per case (t_s, F_N = normal contact force, pen_m = penetration,
z_m, vz_mps), summary.csv (peak force, contact time, max penetration, restitution, impulse), README.txt,
and old_contact_drop.zip of all of it.

Columns of summary.csv: mass_kg, radius_m (aluminum sphere), h_m, v_in_mps, phase (contact start within a step), Fpk_N, contact_s (impact phase: first contact to separation, or to F <= 1.1 m g after the peak when the ball comes to rest), pen_max_m, impulse_Ns (sum F dt over the impact phase), v_out_mps, restitution (v_out / v_in; 0 if the ball rests), rests, n_steps_loaded.
Curves: one row per 2 ms step: t_s, F_N, pen_m (radius - centre height), z_m, vz_mps.
