import os
import shutil
from unittest import TestCase
from mmpreprocesspy.preproc_fun import PreprocessingRunner


class TestPreproc_fun(TestCase):
    test_data_base_path = '/media/micha/T7/20210816_test_data_michael/00_preprocessing_test_data/MM_Testing'

    def test__35__lis__20220320__repeat(self):
        from mmpreprocesspy import preproc_fun
        self.test_data_base_path = "/media/micha/T7/20210816_test_data_michael/00_preprocessing_test_data/MM_Testing/"
        data_directory = self.test_data_base_path + '/35__lis__20220320__repeat/MMSTACK/20220320_VNG1040_AB2h_1_Frame0-478_resaved/'
        directory_to_save = self.test_data_base_path + '/35__lis__20220320__repeat/result_without_flatfield/'
        gl_detection_template_path = self.test_data_base_path + '/35__lis__20220320__repeat/TEMPLATE_MICHAEL_v002/template_config.json'
        image_registration_method = 1
        positions = [0]
        # minframe = 0
        maxframe = 2
        dark_noise = 90
        gaussian_sigma = 5
        # main_channel_angle = -90.7
        # main_channel_angle = 180
        main_channel_angle = 90.1
        normalization_config_path = 'True'
        normalization_region_offset = 120
        frames_to_ignore = []

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        runner = PreprocessingRunner()
        runner.preproc_fun(data_directory, directory_to_save, positions,
                           # minframe=minframe,
                           maxframe=maxframe,
                           dark_noise=dark_noise,
                           gaussian_sigma=gaussian_sigma,
                           main_channel_angle=main_channel_angle,
                           gl_detection_template_path=gl_detection_template_path,
                           normalization_config_path=normalization_config_path,
                           image_registration_method=image_registration_method,
                           normalization_region_offset=normalization_region_offset,
                           frames_to_ignore=frames_to_ignore)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_21_no_flatfield')

    def test__34__nevil__20220302(self):
        from mmpreprocesspy import preproc_fun
        data_directory = self.test_data_base_path + '/34__nevil__20220302/MMStack/202200302_glc_spcm_stress_1/'
        directory_to_save = self.test_data_base_path + '/34__nevil__20220302/result_without_flatfield/'
        gl_detection_template_path = self.test_data_base_path + '/34__nevil__20220302/TEMPLATE/template_config.json'
        image_registration_method = 1
        positions = [0]
        # minframe = 0
        maxframe = 2
        dark_noise = 90
        gaussian_sigma = 5
        # main_channel_angle = -90.7
        # main_channel_angle = 180
        main_channel_angle = 90.6
        normalization_config_path = 'True'
        normalization_region_offset = 120
        frames_to_ignore = []

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        runner = PreprocessingRunner()
        runner.preproc_fun(data_directory, directory_to_save, positions,
                           # minframe=minframe,
                           maxframe=maxframe,
                           dark_noise=dark_noise,
                           gaussian_sigma=gaussian_sigma,
                           main_channel_angle=main_channel_angle,
                           gl_detection_template_path=gl_detection_template_path,
                           normalization_config_path=normalization_config_path,
                           image_registration_method=image_registration_method,
                           normalization_region_offset=normalization_region_offset,
                           frames_to_ignore=frames_to_ignore)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_21_no_flatfield')

    def test__33__lis__20220320(self):
        from mmpreprocesspy import preproc_fun
        data_directory = self.test_data_base_path + '/33__lis__20220320/MMStack/20220320_VNG1040_AB2h_1/'
        directory_to_save = self.test_data_base_path + '/33__lis__20220320/result_without_flatfield/'
        gl_detection_template_path = self.test_data_base_path + '/33__lis__20220320/TEMPLATE/template_config.json'
        image_registration_method = 1
        positions = [0]
        # minframe = 0
        maxframe = 5
        dark_noise = 90
        gaussian_sigma = 5
        # main_channel_angle = -90.7
        # main_channel_angle = 180
        main_channel_angle = 90
        normalization_config_path = 'True'
        normalization_region_offset = 120
        frames_to_ignore = []

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        runner = PreprocessingRunner()
        runner.preproc_fun(data_directory, directory_to_save, positions,
                           # minframe=minframe,
                           maxframe=maxframe,
                           dark_noise=dark_noise,
                           gaussian_sigma=gaussian_sigma,
                           main_channel_angle=main_channel_angle,
                           gl_detection_template_path=gl_detection_template_path,
                           normalization_config_path=normalization_config_path,
                           image_registration_method=image_registration_method,
                           normalization_region_offset=normalization_region_offset,
                           frames_to_ignore=frames_to_ignore)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_21_no_flatfield')

    def test__32__lis__20220128(self):
        from mmpreprocesspy import preproc_fun
        data_directory = self.test_data_base_path + '/32__lis__20220128/MMStack/'
        directory_to_save = self.test_data_base_path + '/32__lis__20220128/result_without_flatfield/'
        gl_detection_template_path = self.test_data_base_path + '/32__lis__20220128/TEMPLATE/template_config.json'
        image_registration_method = 1
        positions = [0]
        # minframe = 0
        maxframe = 5
        dark_noise = 90
        gaussian_sigma = 5
        # main_channel_angle = -90.7
        # main_channel_angle = 180
        main_channel_angle = 90
        normalization_config_path = 'True'
        normalization_region_offset = 120
        frames_to_ignore = []

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        runner = PreprocessingRunner()
        runner.preproc_fun(data_directory, directory_to_save, positions,
                           # minframe=minframe,
                           maxframe=maxframe,
                           dark_noise=dark_noise,
                           gaussian_sigma=gaussian_sigma,
                           main_channel_angle=main_channel_angle,
                           gl_detection_template_path=gl_detection_template_path,
                           normalization_config_path=normalization_config_path,
                           image_registration_method=image_registration_method,
                           normalization_region_offset=normalization_region_offset,
                           frames_to_ignore=frames_to_ignore)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_21_no_flatfield')

    def test__31__dany__20190515(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/31__dany__20190515/MMStack/'
        directory_to_save = self.test_data_base_path + '/31__dany__20190515/result_without_flatfield/'
        gl_detection_template_path = self.test_data_base_path + '/31__dany__20190515/TEMPLATE/template_config.json'
        positions = [19]
        # minframe = 0
        maxframe = 5
        dark_noise = 90
        gaussian_sigma = 5
        # main_channel_angle = -90.7
        # main_channel_angle = 180
        main_channel_angle = -0.5
        normalization_config_path = 'True'
        normalization_region_offset = 90
        frames_to_ignore = []

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        runner = PreprocessingRunner()
        runner.preproc_fun(data_directory, directory_to_save, positions,
                           # minframe=minframe,
                           maxframe=maxframe,
                           dark_noise=dark_noise,
                           gaussian_sigma=gaussian_sigma,
                           main_channel_angle=main_channel_angle,
                           gl_detection_template_path=gl_detection_template_path,
                           normalization_config_path=normalization_config_path,
                           image_registration_method=2,
                           normalization_region_offset=normalization_region_offset,
                           frames_to_ignore=frames_to_ignore)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_21_no_flatfield')

    def test__30__lis_20210813__ignore_frames(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/30__lis_20210813/MMStack/'
        directory_to_save = self.test_data_base_path + '/30__lis_20210813/result_without_flatfield/'
        gl_detection_template_path = self.test_data_base_path + '/30__lis_20210813/TEMPLATE/template_config.json'
        positions = [0]
        # minframe = 0
        maxframe = 5
        dark_noise = 90
        gaussian_sigma = 5
        # main_channel_angle = -90.7
        # main_channel_angle = 180
        main_channel_angle = 90.3
        normalization_config_path = 'True'
        normalization_region_offset = 120
        frames_to_ignore = [1, 3]

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        runner = PreprocessingRunner()
        runner.preproc_fun(data_directory, directory_to_save, positions,
                           # minframe=minframe,
                           maxframe=maxframe,
                           dark_noise=dark_noise,
                           gaussian_sigma=gaussian_sigma,
                           main_channel_angle=main_channel_angle,
                           gl_detection_template_path=gl_detection_template_path,
                           normalization_config_path=normalization_config_path,
                           image_registration_method=2,
                           normalization_region_offset=normalization_region_offset,
                           frames_to_ignore=frames_to_ignore)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_21_no_flatfield')

    def test__30__lis_20210813(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/30__lis_20210813/MMStack/'
        directory_to_save = self.test_data_base_path + '/30__lis_20210813/result_without_flatfield/'
        gl_detection_template_path = self.test_data_base_path + '/30__lis_20210813/TEMPLATE/template_config.json'
        positions = [0]
        # minframe = 0
        maxframe = 5
        dark_noise = 90
        gaussian_sigma = 5
        # main_channel_angle = -90.7
        # main_channel_angle = 180
        main_channel_angle = 90.3
        normalization_config_path = 'True'
        normalization_region_offset = 120

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        runner = PreprocessingRunner()
        runner.preproc_fun(data_directory, directory_to_save, positions,
                                # minframe=minframe,
                                maxframe=maxframe,
                                dark_noise=dark_noise,
                                gaussian_sigma=gaussian_sigma,
                                main_channel_angle=main_channel_angle,
                                gl_detection_template_path=gl_detection_template_path,
                                normalization_config_path=normalization_config_path,
                                image_registration_method=2,
                                normalization_region_offset=normalization_region_offset)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_21_no_flatfield')

    def test__dataset_29_without_flatfield(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/29__dany__20210723_aceglyaa_pl/MMStack/'
        directory_to_save = self.test_data_base_path + '/29__dany__20210723_aceglyaa_pl/result_without_flatfield/'
        gl_detection_template_path = self.test_data_base_path + '/29__dany__20210723_aceglyaa_pl/TEMPLATE/template_config.json'
        positions = [0]
        # minframe = 0
        maxframe = 5
        dark_noise = 90
        gaussian_sigma = 5
        # main_channel_angle = -90.7
        # main_channel_angle = 180  # CODE RUNS, BUT THE IMAGE HAS ROWS REPEATED AT TOP AND BOTTOM
        main_channel_angle = 0
        normalization_config_path = 'True'
        normalization_region_offset = 20

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions,
                                # minframe=minframe,
                                maxframe=maxframe,
                                dark_noise=dark_noise,
                                gaussian_sigma=gaussian_sigma,
                                main_channel_angle=main_channel_angle,
                                gl_detection_template_path=gl_detection_template_path,
                                normalization_config_path=normalization_config_path,
                                image_registration_method=2,
                                normalization_region_offset=normalization_region_offset)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_21_no_flatfield')

    def test__dataset_28_without_flatfield(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/28__theo__20210506/MMStack/'
        directory_to_save = self.test_data_base_path + '/28__theo__20210506/result_without_flatfield/'
        # gl_detection_template_path = self.test_data_base_path + '/28__theo__20210506/DONT_DELETE_gl_detection_template/template_Theo_20210506.json'
        gl_detection_template_path = self.test_data_base_path + '/28__theo__20210506/DONT_DELETE_gl_detection_template/template_Theo_20210506.json'
        positions = [0]
        # minframe = 0
        maxframe = 10
        dark_noise = 90
        gaussian_sigma = 5
        # main_channel_angle = -90.7
        main_channel_angle = 89.3
        normalization_config_path = 'True'
        normalization_region_offset = 20

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions,
                                # minframe=minframe,
                                maxframe=maxframe,
                                dark_noise=dark_noise,
                                gaussian_sigma=gaussian_sigma,
                                main_channel_angle=main_channel_angle,
                                gl_detection_template_path=gl_detection_template_path,
                                normalization_config_path=normalization_config_path,
                                image_registration_method=2,
                                normalization_region_offset=normalization_region_offset)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_21_no_flatfield')

    def test__dataset_26__Lis__20210304_defocus_stack_z_split(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/26__Lis__20210304_defocus_stack/MMStack_z_plane_split/'
        directory_to_save = self.test_data_base_path + '/26__Lis__20210304_defocus_stack/output/'
        gl_detection_template_path = self.test_data_base_path + '/26__Lis__20210304_defocus_stack/GL_DETECTION_TEMPLATE/template_config.json'
        flatfield_directory = os.path.join(self.test_data_base_path, '26__Lis__20210304_defocus_stack/FLATFILED/')
        positions = [0]
        # minframe = 0
        maxframe = 2
        dark_noise = 90
        gaussian_sigma = 5
        main_channel_angle = 89.8
        normalization_config_path = 'True'
        z_slice_index = 0

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions,
                                # minframe=minframe,
                                maxframe=maxframe,
                                dark_noise=dark_noise,
                                gaussian_sigma=gaussian_sigma,
                                main_channel_angle=main_channel_angle,
                                gl_detection_template_path=gl_detection_template_path,
                                normalization_config_path=normalization_config_path,
                                z_slice_index=z_slice_index)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_21_no_flatfield')

    def test__dataset_26__Lis__20210304_defocus_stack(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/26__Lis__20210304_defocus_stack/MMStack/'
        directory_to_save = self.test_data_base_path + '/26__Lis__20210304_defocus_stack/output/'
        gl_detection_template_path = self.test_data_base_path + '/26__Lis__20210304_defocus_stack/GL_DETECTION_TEMPLATE/template_config.json'
        flatfield_directory = os.path.join(self.test_data_base_path, '26__Lis__20210304_defocus_stack/FLATFILED/')
        positions = [0]
        # minframe = 0
        maxframe = 2
        dark_noise = 90
        gaussian_sigma = 5
        main_channel_angle = 89.8
        normalization_config_path = 'True'
        z_slice_index = 7

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions,
                                # minframe=minframe,
                                maxframe=maxframe,
                                dark_noise=dark_noise,
                                gaussian_sigma=gaussian_sigma,
                                main_channel_angle=main_channel_angle,
                                gl_detection_template_path=gl_detection_template_path,
                                normalization_config_path=normalization_config_path,
                                flatfield_directory=flatfield_directory,
                                z_slice_index=z_slice_index)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_21_no_flatfield')

    def test__dataset_25_without_flatfield(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/25__dany__20210423/MMStack/'
        directory_to_save = self.test_data_base_path + '/25__dany__20210423/result_with_flatfield/'
        gl_detection_template_path = self.test_data_base_path + '/25__dany__20210423/DONT_DELETE_gl_detection_template/20210427_preprocessing_template/pos0.json'
        flatfield_directory = os.path.join(self.test_data_base_path, '25__dany__20210423/flatfield')
        positions = [0]
        # minframe = 0
        maxframe = 2
        dark_noise = 90
        gaussian_sigma = 5
        main_channel_angle = -90.7
        normalization_config_path = 'True'

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions,
                                # minframe=minframe,
                                maxframe=maxframe,
                                dark_noise=dark_noise,
                                gaussian_sigma=gaussian_sigma,
                                main_channel_angle=main_channel_angle,
                                gl_detection_template_path=gl_detection_template_path,
                                normalization_config_path=normalization_config_path,
                                flatfield_directory=flatfield_directory)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_21_no_flatfield')

    def test__dataset_24_with_flatfield(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/24__lis__20210303/MMStack/'
        directory_to_save = self.test_data_base_path + '/24__lis__20210303/result_with_flatfield/'
        gl_detection_template_path = self.test_data_base_path + '/24__lis__20210303/DONT_DELETE_gl_detection_template/gl_detection_template_size_1__v004.json'
        flatfield_directory = os.path.join(self.test_data_base_path, '24__lis__20210303/flatfield__lis__20210211')
        positions = [21]
        maxframe = 2
        dark_noise = 90
        gaussian_sigma = 5
        main_channel_angle = 89.7
        normalization_config_path = 'True'

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions, maxframe=maxframe, dark_noise=dark_noise,
                                gaussian_sigma=gaussian_sigma,
                                main_channel_angle=main_channel_angle,
                                gl_detection_template_path=gl_detection_template_path,
                                normalization_config_path=normalization_config_path,
                                flatfield_directory=flatfield_directory)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_21_no_flatfield')

    def test__dataset_23_no_flatfield(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/23__thomas__20170403_synthPromsHi_glc_aMG_1/MMStack/'
        directory_to_save = self.test_data_base_path + '/23__thomas__20170403_synthPromsHi_glc_aMG_1/MMStack/result_no_flatfield/'
        gl_detection_template_path = self.test_data_base_path + '/23__thomas__20170403_synthPromsHi_glc_aMG_1/DONT_DELETE_gl_detection_template/gl_detection_template.json'
        positions = [2]
        maxframe = 1
        dark_noise = 90
        gaussian_sigma = 5
        main_channel_angle = 0.9
        # main_channel_angle = None
        growthlane_length_threshold = 100

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions, maxframe=maxframe, dark_noise=dark_noise,
                                growthlane_length_threshold=growthlane_length_threshold, gaussian_sigma=gaussian_sigma,
                                main_channel_angle=main_channel_angle,
                                gl_detection_template_path=gl_detection_template_path)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_21_no_flatfield')

    def test__dataset_20_with_flatfield(self):
        from mmpreprocesspy import preproc_fun

        path = os.path.join(self.test_data_base_path, '20__theo__20210122')
        data_directory = os.path.join(path, '20210122_glu_lac_1')
        flatfield_directory = os.path.join(path, '20210122_flatField')
        output_directory = os.path.join(path, 'output_with_flatfield')

        positions = [0]
        maxframe = 1
        dark_noise = 90
        gaussian_sigma = 5
        main_channel_angle = 90
        # main_channel_angle = None
        growthlane_length_threshold = 200

        if os.path.isdir(output_directory):
            shutil.rmtree(output_directory)

        preproc_fun.preproc_fun(data_directory, output_directory, positions, flatfield_directory=flatfield_directory,
                                maxframe=maxframe, dark_noise=dark_noise,
                                growthlane_length_threshold=growthlane_length_threshold, gaussian_sigma=gaussian_sigma,
                                main_channel_angle=main_channel_angle)

        self.read_and_show_gl_index_image(os.path.join(output_directory, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_20_with_flatfield')

    def test__dataset_19_no_flatfield(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/19__dany__20201123_comlac_3conds_5/MMStack/'
        directory_to_save = self.test_data_base_path + '/19__dany__20201123_comlac_3conds_5/MMStack/result_no_flatfield/'
        positions = [0]
        maxframe = 1
        dark_noise = 90
        gaussian_sigma = 5
        main_channel_angle = 0
        # main_channel_angle = None
        growthlane_length_threshold = 200

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions, maxframe=maxframe, dark_noise=dark_noise,
                                growthlane_length_threshold=growthlane_length_threshold, gaussian_sigma=gaussian_sigma, main_channel_angle=main_channel_angle)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_19_no_flatfield')

    def test__dataset_17_no_flatfield(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/17_lis_20201218_VNG40_AB6min_2h_1_1/MMStack/'
        directory_to_save = self.test_data_base_path + '/17_lis_20201218_VNG40_AB6min_2h_1_1/MMStack/result_no_flatfield/'
        positions = [0]
        maxframe = 1
        dark_noise = 90
        gaussian_sigma = 5
        main_channel_angle = 90
        # main_channel_angle = None
        growthlane_length_threshold = 200

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions, maxframe=maxframe, dark_noise=dark_noise,
                                growthlane_length_threshold=growthlane_length_threshold, gaussian_sigma=gaussian_sigma, main_channel_angle=main_channel_angle)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_17_no_flatfield')


    def test__dataset_16_no_flatfield_using_gl_detection_template(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/16_thomas_20201229_glc_lac_1/MMStack/'
        directory_to_save = self.test_data_base_path + '/16_thomas_20201229_glc_lac_1/MMStack/result_no_flatfield_with_gl_detection_template/'
        gl_detection_template_path = './data/test_preproc_fun/16_thomas_20201229_glc_lac_1.json'
        positions = [0]
        maxframe = 5
        dark_noise = 90
        gaussian_sigma = 5
        growthlane_length_threshold = 300
        normalization_config_path = 'True'

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions, maxframe=maxframe, dark_noise=dark_noise,
                                growthlane_length_threshold=growthlane_length_threshold, gaussian_sigma=gaussian_sigma,
                                gl_detection_template_path=gl_detection_template_path,
                                normalization_config_path=normalization_config_path)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_16_no_flatfield_with_gl_detection_template')

    def test__dataset_16_no_flatfield(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/16_thomas_20201229_glc_lac_1/MMStack/'
        directory_to_save = self.test_data_base_path + '/16_thomas_20201229_glc_lac_1/MMStack/result_no_flatfield/'
        positions = [0]
        maxframe = 1
        dark_noise = 90
        gaussian_sigma = 5
        growthlane_length_threshold = 300

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions, maxframe=maxframe, dark_noise=dark_noise,
                                growthlane_length_threshold=growthlane_length_threshold, gaussian_sigma=gaussian_sigma)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_16_no_flatfield')

    def test__dataset_15_no_flatfield(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/15_lis_20201119_VNG1040_AB2h_2h_1/MMStack/'
        directory_to_save = self.test_data_base_path + '/15_lis_20201119_VNG1040_AB2h_2h_1/MMStack/result_no_flatfield/'
        positions = [1]
        maxframe = 3
        dark_noise = 90
        gaussian_sigma = 5
        main_channel_angle = 90
        # main_channel_angle = None
        growthlane_length_threshold = 300
        roi_boundary_offset_at_mother_cell = 0

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory,
                                directory_to_save,
                                positions,
                                maxframe=maxframe,
                                dark_noise=dark_noise,
                                growthlane_length_threshold=growthlane_length_threshold,
                                gaussian_sigma=gaussian_sigma,
                                main_channel_angle=main_channel_angle,
                                roi_boundary_offset_at_mother_cell=roi_boundary_offset_at_mother_cell)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_15_no_flatfield')

    def test__dataset_14_no_flatfield(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/14_thomas_20201228_glc_ara_1/MMStack/'
        directory_to_save = self.test_data_base_path + '/14_thomas_20201228_glc_ara_1/MMStack/result_no_flatfield/'
        positions = [0]
        maxframe = 1
        dark_noise = 90
        gaussian_sigma = 5
        growthlane_length_threshold = 300

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions, maxframe=maxframe, dark_noise=dark_noise,
                                growthlane_length_threshold=growthlane_length_threshold, gaussian_sigma=gaussian_sigma)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_14_no_flatfield')


    def test__dataset_13_with_flatfield(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/13_20200128_glcIPTG_glc_1/MMStack/RawData/measurement'
        flatfield_directory = self.test_data_base_path + '/13_20200128_glcIPTG_glc_1/MMStack/RawData/flatfield'
        directory_to_save = self.test_data_base_path + '/13_20200128_glcIPTG_glc_1/MMStack/result_with_flatfield/'
        positions = [1]

        minframe = 1
        maxframe = 8
        dark_noise = None
        gaussian_sigma = None
        growthlane_length_threshold = 200
        main_channel_angle = 0

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions, minframe=minframe, maxframe=maxframe,
                                growthlane_length_threshold=growthlane_length_threshold, flatfield_directory=flatfield_directory,
                                main_channel_angle=main_channel_angle, dark_noise=dark_noise, gaussian_sigma=gaussian_sigma)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_13_with_flatfield')

    def test__dataset_10_with_flatfield(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/measurement'
        flatfield_directory = self.test_data_base_path + '/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/flatfield'
        directory_to_save = self.test_data_base_path + '/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/result_with_flatfield/'
        positions = [0]
        minframe = None
        maxframe = 8
        dark_noise = None
        gaussian_sigma = None

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions, minframe=minframe, maxframe=maxframe,
                                growthlane_length_threshold=300, flatfield_directory=flatfield_directory,
                                dark_noise=dark_noise, gaussian_sigma=gaussian_sigma)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_10_with_flatfield')

    def test__dataset_11_with_flatfield(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/11_20190910_glc_spcm_1/MMStack'
        flatfield_directory = self.test_data_base_path + '/11_20190910_glc_spcm_1/MMStack/flatfield'
        directory_to_save = self.test_data_base_path + '/11_20190910_glc_spcm_1/MMStack/result_with_flatfield/'
        positions = [0]
        minframe = None
        maxframe = 8
        dark_noise = None
        gaussian_sigma = None

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions, minframe=minframe, maxframe=maxframe,
                                growthlane_length_threshold=300, flatfield_directory=flatfield_directory,
                                dark_noise=dark_noise, gaussian_sigma=gaussian_sigma)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_11_with_flatfield')

    def test__dataset_11_no_flatfield(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/11_20190910_glc_spcm_1/MMStack'
        directory_to_save = self.test_data_base_path + '/11_20190910_glc_spcm_1/MMStack/result_no_flatfield/'
        positions = [0]
        minframe = None
        maxframe = 8
        dark_noise = 90
        gaussian_sigma = 5

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions, maxframe=maxframe,
                                growthlane_length_threshold=300, dark_noise=dark_noise, gaussian_sigma=gaussian_sigma)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_11_no_flatfield')

    def test__dataset_12_with_flatfield(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/12_20190816_Theo/MMStack/'
        flatfield_directory = self.test_data_base_path + '/12_20190816_Theo/MMStack/flatfield'
        directory_to_save = self.test_data_base_path + '/12_20190816_Theo/MMStack/result_with_flatfield/'
        positions = [0]
        minframe = None
        maxframe = 8
        dark_noise = 90
        gaussian_sigma = 5
        growthlane_length_threshold = 200

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions, minframe=minframe, maxframe=maxframe, flatfield_directory=flatfield_directory, dark_noise=dark_noise,
                                gaussian_sigma=gaussian_sigma,
                                growthlane_length_threshold=growthlane_length_threshold)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_12_with_flatfield')

    def test__dataset_12_no_flatfield_using_gl_detection_template(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/12_20190816_Theo/MMStack/'
        directory_to_save = self.test_data_base_path + '/12_20190816_Theo/MMStack/result_no_flatfield_using_gl_detection_template/'
        gl_detection_template_path = './data/test_preproc_fun/12_theo_20190816_glc_spcm_1_MMStack_2__template_v00.json'
        positions = [0]
        maxframe = 3
        dark_noise = 90
        gaussian_sigma = 5
        growthlane_length_threshold = 200
        normalization_config_path = 'True'

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions, maxframe=maxframe, dark_noise=dark_noise,
                                gaussian_sigma=gaussian_sigma,
                                growthlane_length_threshold=growthlane_length_threshold,
                                gl_detection_template_path=gl_detection_template_path,
                                normalization_config_path=normalization_config_path)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_12_no_flatfield_using_gl_detection_template')

    def test__dataset_12_no_flatfield(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/12_20190816_Theo/MMStack/'
        directory_to_save = self.test_data_base_path + '/12_20190816_Theo/MMStack/result_no_flatfield/'
        positions = [0]
        maxframe = 8
        dark_noise = 90
        gaussian_sigma = 5
        growthlane_length_threshold = 200

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        preproc_fun.preproc_fun(data_directory, directory_to_save, positions, maxframe=maxframe, dark_noise=dark_noise,
                                gaussian_sigma=gaussian_sigma,
                                growthlane_length_threshold=growthlane_length_threshold)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_12_no_flatfield')

    def test__dataset_11(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/11_20190910_glc_spcm_1/MMStack'
        directory_to_save = data_directory
        positions = [0]
        maxframe = 5

        results_directory = directory_to_save + '/result/'
        if os.path.isdir(results_directory):
            shutil.rmtree(results_directory)

        preproc_fun.preproc_fun(data_directory, results_directory, positions=positions, maxframe=maxframe, growthlane_length_threshold=300)

        self.read_and_show_gl_index_image(os.path.join(results_directory, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_11')

    def test__dataset_04(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/04_20180531_gluIPTG5uM_lac_1/MMStack'
        directory_to_save = data_directory
        positions = [0]
        maxframe = 10
        growthlane_length_threshold = 300

        results_directory = directory_to_save + '/result/'
        if os.path.isdir(results_directory):
            shutil.rmtree(results_directory)

        preproc_fun.preproc_fun(data_directory, results_directory, positions=positions, maxframe=maxframe,
                                growthlane_length_threshold=growthlane_length_threshold)

        self.read_and_show_gl_index_image(os.path.join(results_directory, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_04')

    def test__dataset_08(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/08_20190222_LB_SpentLB_TrisEDTA_LB_1/MMStack'
        directory_to_save = data_directory
        positions = [0]
        maxframe = 10
        growthlane_length_threshold = 300

        results_directory = directory_to_save + '/result/'
        if os.path.isdir(results_directory):
            shutil.rmtree(results_directory)

        preproc_fun.preproc_fun(data_directory, results_directory, positions=positions, maxframe=maxframe,
                                growthlane_length_threshold=growthlane_length_threshold)

        self.read_and_show_gl_index_image(os.path.join(results_directory, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_08')

    def test__dataset_10(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/measurement'
        flatfield_directory = self.test_data_base_path + '/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/flatfield'
        directory_to_save = self.test_data_base_path + '/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack'
        positions = [0]
        minframe = None
        maxframe = 8
        dark_noise = 90
        gaussian_sigma = 5
        growthlane_length_threshold = 300

        results_directory = directory_to_save + '/result/'
        if os.path.isdir(results_directory):
            shutil.rmtree(results_directory)

        preproc_fun.preproc_fun(data_directory, results_directory, positions, minframe=minframe, maxframe=maxframe,
                                flatfield_directory=flatfield_directory, dark_noise=dark_noise,
                                gaussian_sigma=gaussian_sigma,
                                growthlane_length_threshold=growthlane_length_threshold)

        self.read_and_show_gl_index_image(os.path.join(results_directory, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_10')

    def test__dataset_10__shift_is_determined_correctly(self):
        # There is jump of the image position in dataset 10 going from frame 603 to 604. This trips up the current
        # current implementation. This test was created for debugging this issue.

        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/measurement'
        flatfield_directory = self.test_data_base_path + '/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/flatfield'
        directory_to_save = self.test_data_base_path + '/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/result'
        positions = [0]
        minframe = 590
        maxframe = 610
        dark_noise = 90
        gaussian_sigma = 5
        growthlane_length_threshold = 300

        results_directory = directory_to_save + '/result/'
        if os.path.isdir(results_directory):
            shutil.rmtree(results_directory)

        preproc_fun.preproc_fun(data_directory, results_directory, positions, minframe=minframe, maxframe=maxframe,
                                flatfield_directory=flatfield_directory,
                                dark_noise=dark_noise,
                                gaussian_sigma=gaussian_sigma,
                                growthlane_length_threshold=growthlane_length_threshold)

    def test__get_gl_tiff_path(self):
        from mmpreprocesspy import preproc_fun

        gl_file_path = preproc_fun.get_gl_tiff_path('/path/to/file', 'experiment_name', '1', '5')
        print(gl_file_path)
        print(os.path.dirname(gl_file_path))


    def read_and_show_gl_index_image(self, path, title=None):
        import tifffile as tff
        image_with_rois = tff.imread(path)
        self.show_image(image_with_rois, title=title)


    def show_image(self, imdata, title=None):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        # axes.plot(xs, ys)
        axes.imshow(imdata, cmap='gray')
        axes.get_xaxis().set_visible(False)
        # fig.tight_layout(pad=0)
        plt.axis('off')
        if title:
            plt.title(title)
        plt.show()


