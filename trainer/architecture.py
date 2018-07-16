
class Architecture:

    img_size = 28
    num_cat = 10

    # Generator layer config
    layers_g = [
        {
            'filters':256,
            'kernel_size':[7,7],
            'strides':[1,1],
            'padding':'valid'
        },
        {
            'filters':128,
            'kernel_size':[5,5],
            'strides':[2,2],
            'padding':'same'
        },
        {
            'filters':1,
            'kernel_size':[5,5],
            'strides':[2,2],
            'padding':'same'
        }
    ]

    # Discriminator layer config
    layers_d = [
        {
            'filters': 128,
            'kernel_size': [5,5],
            'strides':[2,2],
            'padding':'same'
        },
        {
            'filters': 256,
            'kernel_size':[5,5],
            'strides':[2,2],
            'padding':'same'
        },
        {
            'filters': 1,
            'kernel_size':[7,7],
            'strides':[1,1],
            'padding':'valid'
        }
    ]
    

