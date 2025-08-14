from measure_diversity import two_d

def test_two_d():
    points_all = two_d.create_normed_datapoints(n=2)
    two_d.plot_points(points_all, title="2 Uniform Points Around Circle")
    points_first_quad = two_d.create_normed_datapoints(n=2, quads=[1])
    two_d.plot_points(points_first_quad, title="2 Points in First Quadrant of Circle")
    points_first_and_third_quad = two_d.create_normed_datapoints(n=2, quads=[1, 3])
    two_d.plot_points(points_first_and_third_quad, title="2 Points in First and Third Quadrants of Circle")
    points_second_quad = two_d.create_normed_datapoints(n=2, quads=[2])
    two_d.plot_points(points_second_quad, title="2 Points in Second Quadrant of Circle")
