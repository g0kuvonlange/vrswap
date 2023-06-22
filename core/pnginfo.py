from PIL import Image as PILImage


def load_pnginfo(image_path):
    target_image = PILImage.open(image_path)

    theta_str = target_image.text.get('Theta')
    phi_str = target_image.text.get('Phi')

    if theta_str and phi_str:
        theta = float(theta_str)
        phi = float(phi_str)
        return theta, phi
    else:
        return None, None
