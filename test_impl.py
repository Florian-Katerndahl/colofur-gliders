from colorful_gliders.PointCloud import PointCloud

pc = PointCloud()
pc.load_from_laz("data/chair-final.laz")
pc.write_to_laz("data/out.laz")

pc.construct_mesh()
pc.write_mesh("data/model.ply")