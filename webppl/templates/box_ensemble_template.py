BOX_ENSEMBLE_TEMPLATE = {
    0 : """
        var boxes = generate_boxes_from_arr(
            [
                [190, 325-20],
                [178, 325-60],
                [165, 325-100],
                [340, 325-40],
                [320, 325-100],
                [395, 325-140],
                [270, 325-160],
                [405, 325-180],
            ],
            [
                [20, 20],
                [20, 20],
                [20, 20],
                [20, 40],
                [80, 20],
                [20, 20],
                [20, 40],
                [20, 20],
            ],
            [
                'red',
                'red',
                'red',
                'purple',
                'purple',
                'red',
                'purple',
                'red',
        ], noise_pos, noise_size, [])
        """,
    1 : """
        var boxes = generate_boxes_from_arr(
            [
                [220, 325-20],
                [340, 325-40],
                [320, 325-100],
                [375, 325-140],
                [265, 325-180],
                [245, 325-140],
                [245, 325-220],
                [395, 325-200],
            ],
            [
                [20, 20],
                [20, 40],
                [80, 20],
                [20, 20],
                [20, 20],
                [20, 20],
                [20, 20],
                [20, 40],
            ],
            [
                'purple',
                'purple',
                'red',
                'red',
                'purple',
                'purple',
                'purple',
                'red',
            ], noise_pos, noise_size, [])
        """,
    2 : """
        var boxes = generate_boxes_from_arr(
      [
        [250, 325 - 60],
        [300, 325 - 140],
        [340, 325 - 27],
        [335, 325 - 81],
        [250, 325 - 190],
        [350, 325 - 190],
        [250, 325 - 250],
        [350, 325 - 250],
      ], 
      [
        [20,60],
        [65,20],
        [20, 27],
        [20, 27],
        [20, 30],
        [20, 30],
        [20, 30],
        [20, 30],
      ],
      [
        'red',
        'purple',
        'purple',
        'purple',
        'purple',
        'red',
        'purple',
        'red',
      ], noise_pos, noise_size, []) 
        """,  #TODO: fix colors on this one
    3 : """var boxes = generate_boxes_from_arr(
            [
                [260, 325-20],
                [340, 325-20],
                [300, 325-60],
                [260, 325-100],
                [340, 325-100],
                [300, 325-140],
                [301, 325-200],
                [299, 325-280],
            ],
            [
                [40, 20],
                [40, 20],
                [40, 20],
                [40, 20],
                [40, 20],
                [40, 20],
                [5, 40],
                [5, 40],
            ],
            [
                'purple',
                'red',
                'purple',
                'purple',
                'red',
                'purple',
                'red',
                'red',
        ], noise_pos, noise_size, [])""",

    4 : """var boxes = generate_boxes_from_arr(
            [
                [200, 325-20],
                [400, 325-20],
                [360, 325-60],
                [360, 325-130],
                [285, 325-40],
                [265, 325-120],
                [285, 325-200],
                [265, 325-280]
            ],
            [
                [20, 20],
                [20, 20],
                [20, 60],
                [60, 10],
                [20, 40],
              [20, 40],
              [20, 40],
              [20, 40],
            ],
            [
                'purple',
                'purple',
                'purple',
                'purple',
                'red',
                'red',
                'red',
                'red',
        ], noise_pos, noise_size, [])""",

    5 : """
    var boxes = generate_boxes_from_arr(
            [
                [220, 325-20],
              [240, 325-60],
              [220, 325-100],
              [240, 325-140],
              [380, 325-20],
              [360, 325-60],
              [380, 325-100],
              [360, 325-140],
            ],
            [
                [20, 20],
              [40, 20],
              [20, 20],
              [40, 20],
              [20, 20],
              [40, 20],
              [20, 20],
              [40, 20],
            ],
            [
                'purple',
                'purple',
                'purple',
                'purple',
                'red',
                'red',
                'red',
                'red',
        ], noise_pos, noise_size, [])""",
    6 : """var boxes = generate_boxes_from_arr(
            [
                [390, 325-30],
                [390, 325-90],
                [390, 325-150],
                [250, 325-40],
                [270, 325-100],
                [280, 325-160],
                [260, 325-220],
                [275, 325-270],
            ],
            [
                [20, 30],
                [20, 30],
                [20, 30],
                [20, 40],
                [60, 20],
                [20, 40],
                [60, 20],
                [20, 30]
            ],
            [
                'purple',
                'purple',
                'purple',
                'red',
                'red',
                'red',
                'red',
                'red',
        ], noise_pos, noise_size, [])""",

    7 : """

    var boxes = generate_boxes_from_arr(
            [
                [260, 325-20],
                [340, 325-20],
                [300, 325-65],
                [320, 325 - 110],
                [360, 325 - 110],
                [340, 325 - 150],
                [250, 325 - 110],
                [265, 325 - 160],
            ],
            [
                [20, 20],
                [20, 20],
                [60, 25],
                [20, 20],
                [20, 20],
                [20, 20],
                [20, 20],
                [30, 30]
            ],
            [
                'purple',
                'purple',
                'red',
                'purple',
                'purple',
                'red',
                'red',
                'red',
        ], noise_pos, noise_size, [])
        """
}