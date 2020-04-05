from __future__ import absolute_import

__version__ = "0.1.1"

from psenet_text_detector.utils import (read_image,
                                        export_detected_regions,
                                        visualize_detection)

#from psenet_text_detector.utils import (export_detected_regions,
#                                            export_extra_results)

from psenet_text_detector.predict import (load_psenet_model,
                                          get_prediction)

# load craft model
psenet = load_psenet_model()


# detect texts
def detect_text(image_path,
                output_dir,
                cuda=False,
                export_extra=True,
                binary_th=1.0,
                kernel_num=3,
                upsample_scale=1,
                long_size=1280,
                min_kernel_area=10.0,
                min_area=300.0,
                min_score=0.93):
    """
    Arguments:
        image_path: path to the image to be processed
        output_dir: path to the results to be exported
        cuda: Use cuda for inference
        export_extra: export box visualization
    Output:
        {"boxes": list of coords of points of predicted boxes}
    """
    # load image
    image = read_image(image_path)

    # perform prediction
    prediction_result = get_prediction(image=image,
                                       model=psenet,
                                       binary_th=binary_th,
                                       kernel_num=kernel_num,
                                       upsample_scale=upsample_scale,
                                       long_size=long_size,
                                       min_kernel_area=min_kernel_area,
                                       min_area=min_area,
                                       min_score=min_score,
                                       cuda=cuda)

    # export if output_dir is given
    prediction_result["text_crop_paths"] = []
    if output_dir is not None:
        # export detected regions
        exported_file_paths = export_detected_regions(image_path,
                                                      image,
                                                      boxes=prediction_result["boxes"],
                                                      output_dir=output_dir)

        prediction_result["text_crop_paths"] = exported_file_paths

        # export box visualization
        if export_extra:
            _ = visualize_detection(image_path,
                                    image=image,
                                    quads=prediction_result["boxes"],
                                    output_dir=output_dir)

    # return prediction results
    return prediction_result
