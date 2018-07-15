
class Architecture:

    # Generator layer config
    layers_g = [
        {
            'filters':256,
            'kernel_size':[4,4],
            'strides':[1,1],
            'padding':'valid'
        },
        {
            'filters':128,
            'kernel_size':[4,4],
            'strides':[2,2],
            'padding':'same'
        },
        {
            'filters':64,
            'kernel_size':[4,4],
            'strides':[2,2],
            'padding':'same'
        },
        {
            'filters':1,
            'kernel_size':[4,4],
            'strides':[2,2],
            'padding':'same'
        }
    ]

    # Discriminator layer config
    layers_d = [
        {
            'filters': 64,
            'kernel_size': [4,4],
            'strides':[2,2],
            'padding':'same'
        },
        {
            'filters': 128,
            'kernel_size':[4,4],
            'strides':[2,2],
            'padding':'same'
        },
        {
            'filters': 256,
            'kernel_size':[4,4],
            'strides':[2,2],
            'padding':'same'
        },
        {
            'filters': 1,
            'kernel_size':[4,4],
            'strides':[1,1],
            'padding':'valid'
        }
    ]
    

