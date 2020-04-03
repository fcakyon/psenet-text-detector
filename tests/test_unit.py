import unittest
import psenet_text_detector as pse

image_path = 'figures/idcard.png'


class TestTextDetector(unittest.TestCase):
    def test_load_craftnet_model(self):
        psenet = pse.load_psenet_model()
        self.assertTrue(psenet)

    def test_get_prediction(self):
        # load image
        image = pse.read_image(image_path)

        # load models
        psenet = pse.load_psenet_model()

        # perform prediction
        get_prediction = pse.get_prediction
        prediction_result = get_prediction(image=image,
                                           model=psenet,
                                           cuda=False,
                                           binary_th=1.0,
                                           kernel_num=3,
                                           upsample_scale=1,
                                           long_size=1280,
                                           min_kernel_area=10.0,
                                           min_area=300.0,
                                           min_score=0.93)

        self.assertEqual(len(prediction_result["boxes"]), 37)
        self.assertEqual(len(prediction_result["boxes"][0]), 4)
        self.assertEqual(len(prediction_result["boxes"][0][0]), 2)
        self.assertEqual(int(prediction_result["boxes"][0][0][0]), 239)

    def test_detect_text(self):
        prediction_result = pse.detect_text(image_path,
                                            output_dir=None,
                                            cuda=False,
                                            export_extra=False,
                                            binary_th=1.0,
                                            kernel_num=3,
                                            upsample_scale=1,
                                            long_size=1280,
                                            min_kernel_area=10.0,
                                            min_area=300.0,
                                            min_score=0.93)
        self.assertEqual(len(prediction_result["boxes"]), 37)
        self.assertEqual(len(prediction_result["boxes"][0]), 4)
        self.assertEqual(len(prediction_result["boxes"][0][0]), 2)
        self.assertEqual(int(prediction_result["boxes"][0][0][0]), 239)

        prediction_result = pse.detect_text(image_path,
                                            output_dir=None,
                                            cuda=False,
                                            export_extra=False,
                                            binary_th=1.0,
                                            kernel_num=1,
                                            upsample_scale=1,
                                            long_size=800,
                                            min_kernel_area=5.0,
                                            min_area=300.0,
                                            min_score=0.93)
        self.assertEqual(len(prediction_result["boxes"]), 23)
        self.assertEqual(len(prediction_result["boxes"][0]), 4)
        self.assertEqual(len(prediction_result["boxes"][0][0]), 2)
        self.assertEqual(int(prediction_result["boxes"][0][0][0]), 238)


if __name__ == '__main__':
    unittest.main()
