def upscale_bbox():
    print("Load frame")
    frame = cv2.imread("D:\\test\\input.jpg")
    target_face = get_face(frame)

    print("Slicing Bounding box")
    x1, y1, x2, y2 = map(int, target_face['bbox']) # gets coordinates from insight face analyser

    # Add a slight outer margin to the bounding box and crop it
    h = y2 - y1
    w = x2 - x1
    y1 = max(0, y1 - int(0.05 * h))
    x1 = max(0, x1 - int(0.05 * w))
    y2 = min(frame.shape[0], y2 + int(0.05 * h))
    x2 = min(frame.shape[1], x2 + int(0.05 * w))

    bbox_image = frame[y1:y2, x1:x2]
    cv2.imwrite("D:\\test\\face.png", bbox_image)
    print("Saved to D:\\test\\face.png")

    print("Run codeformer")
    codeformer_command = f'python CodeFormer/inference_codeformer.py -i "D:\\test\\face.png" -o "D:\\test\\face" -w 0.85 -s 1 --face_upsample'
    subprocess.run(codeformer_command, shell=True, check=True)

    print("Integrate face back in frame")
    upscaled_image = cv2.imread("D:\\test\\face\\final_results\\face.png")
    upscaled_h, upscaled_w, _ = upscaled_image.shape
    bbox_h, bbox_w, _ = bbox_image.shape
    upscaled_image_resized = cv2.resize(upscaled_image, (bbox_w-2, bbox_h-2))

    # Create a mask of the same size as the upscaled image
    mask = 255 * np.ones(upscaled_image_resized.shape, upscaled_image_resized.dtype)

    # The center of the cloned area will be the center of the bounding box
    center = ((x1+x2)//2, (y1+y2)//2)

    # Seamless cloning to blend the upscaled image with the original frame
    output = cv2.seamlessClone(upscaled_image_resized, frame, mask, center, cv2.MIXED_CLONE)

    cv2.imwrite("D:\\test\\final_output.jpg", output)

    return True