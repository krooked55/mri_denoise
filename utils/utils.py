def show_windows_image(gt, noise, denoise):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from matplotlib.patches import Rectangle

    # Lista de imágenes
    show_gt = []
    show_noise = []
    show_denoise = []
    for i in range(5):
        image_gt = gt[i].reshape(gt.shape[1], gt.shape[1])
        image_noise = noise[i].reshape(gt.shape[1], gt.shape[1])
        image_denoise = denoise[i].reshape(gt.shape[1], gt.shape[1])

        show_gt.append(image_gt)
        show_noise.append(image_noise)
        show_denoise.append(image_denoise)

    show_better_images = []
    for i in [show_gt, show_noise, show_denoise]:
        show_better_images.extend(i)
    ################


    # Configurar el tamaño de la figura y los ejes
    fig, ax = plt.subplots(3, 5, figsize=(15, 9))

    # Iterar a través de las imágenes y mostrarlas en los ejes correspondientes
    for i, imagen in enumerate(show_better_images):
        fila = i // 5  # Número de fila
        columna = i % 5  # Número de columna
        ax[fila, columna].imshow(imagen, cmap='gray')
        ax[fila, columna].axis('off')

        #CÓDIGO ZOOM QUE FUNCIONA DEFINITIVO
        # Definir las coordenadas de la región de interés (ROI)
        x1, y1, x2, y2 = 80, 80, 120, 120

        # Crear los ejes para el recuadro
        axins = ax[fila, columna].inset_axes([0, 0, 0.4, 0.4]) #x1, x2, %ancho, %alto
        axins.imshow(imagen)
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.axis('off')

        # Dibujar el recuadro en la figura principal
        #mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="red")
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', linewidth=2)
        ax[fila, columna].add_patch(rect)

        ax[fila, columna].axis("off")

        rect = Rectangle((0, 0), 40*160/100, 40*160/100, edgecolor='red', facecolor='none', linewidth=3)
        ax[fila, columna].add_patch(rect)
        ax[fila, columna].invert_yaxis()
        ###########################################

    # Ajustar el espacio entre las imágenes
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()


def show_image(gt, noise, denoise):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from matplotlib.patches import Rectangle

    # Lista de imágenes
    show_gt = []
    show_noise = []
    show_denoise = []
    for i in range(5):
        image_gt = gt[i].reshape(gt.shape[1], gt.shape[1])
        image_noise = noise[i].reshape(gt.shape[1], gt.shape[1])
        image_denoise = denoise[i].reshape(gt.shape[1], gt.shape[1])

        show_gt.append(image_gt)
        show_noise.append(image_noise)
        show_denoise.append(image_denoise)

    show_better_images = []
    for i in [show_gt, show_noise, show_denoise]:
        show_better_images.extend(i)
    ################


    # Configurar el tamaño de la figura y los ejes
    fig, ax = plt.subplots(3, 5, figsize=(15, 9))

    # Iterar a través de las imágenes y mostrarlas en los ejes correspondientes
    for i, imagen in enumerate(show_better_images):
        fila = i // 5  # Número de fila
        columna = i % 5  # Número de columna
        ax[fila, columna].imshow(imagen, cmap='gray')
        ax[fila, columna].axis('off')
    # Ajustar el espacio entre las imágenes
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()


def show_windows_image_worstbest(gt, noise, denoise, worst_best):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from matplotlib.patches import Rectangle

    # Lista de imágenes
    show_gt = []
    show_noise = []
    show_denoise = []
    for i in worst_best:
        image_gt = gt[i].reshape(gt.shape[1], gt.shape[1])
        image_noise = noise[i].reshape(gt.shape[1], gt.shape[1])
        image_denoise = denoise[i].reshape(gt.shape[1], gt.shape[1])

        show_gt.append(image_gt)
        show_noise.append(image_noise)
        show_denoise.append(image_denoise)

    show_better_images = []
    for i in [show_gt, show_noise, show_denoise]:
        show_better_images.extend(i)
    ################


    # Configurar el tamaño de la figura y los ejes
    fig, ax = plt.subplots(3, 5, figsize=(15, 9))

    # Iterar a través de las imágenes y mostrarlas en los ejes correspondientes
    for i, imagen in enumerate(show_better_images):
        fila = i // 5  # Número de fila
        columna = i % 5  # Número de columna
        ax[fila, columna].imshow(imagen, cmap='gray')
        ax[fila, columna].axis('off')

        #CÓDIGO ZOOM QUE FUNCIONA DEFINITIVO
        # Definir las coordenadas de la región de interés (ROI)
        x1, y1, x2, y2 = 80, 80, 120, 120

        # Crear los ejes para el recuadro
        axins = ax[fila, columna].inset_axes([0, 0, 0.4, 0.4]) #x1, x2, %ancho, %alto
        axins.imshow(imagen)
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.axis('off')

        # Dibujar el recuadro en la figura principal
        #mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="red")
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', linewidth=2)
        ax[fila, columna].add_patch(rect)

        ax[fila, columna].axis("off")

        rect = Rectangle((0, 0), 40*160/100, 40*160/100, edgecolor='red', facecolor='none', linewidth=3)
        ax[fila, columna].add_patch(rect)
        ax[fila, columna].invert_yaxis()
        ###########################################

    # Ajustar el espacio entre las imágenes
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()


def show_image_worstbest(gt, noise, denoise, worst_best):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from matplotlib.patches import Rectangle

    # Lista de imágenes
    show_gt = []
    show_noise = []
    show_denoise = []
    for i in worst_best:
        image_gt = gt[i].reshape(gt.shape[1], gt.shape[1])
        image_noise = noise[i].reshape(gt.shape[1], gt.shape[1])
        image_denoise = denoise[i].reshape(gt.shape[1], gt.shape[1])

        show_gt.append(image_gt)
        show_noise.append(image_noise)
        show_denoise.append(image_denoise)

    show_better_images = []
    for i in [show_gt, show_noise, show_denoise]:
        show_better_images.extend(i)
    ################


    # Configurar el tamaño de la figura y los ejes
    fig, ax = plt.subplots(3, 5, figsize=(15, 9))

    # Iterar a través de las imágenes y mostrarlas en los ejes correspondientes
    for i, imagen in enumerate(show_better_images):
        fila = i // 5  # Número de fila
        columna = i % 5  # Número de columna
        ax[fila, columna].imshow(imagen, cmap='gray')
        ax[fila, columna].axis('off')
    # Ajustar el espacio entre las imágenes
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()