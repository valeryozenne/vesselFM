import SimpleITK as sitk
import os
from .utils import save_array, save_metadata, calculate_metadata, convert_sitk_image, resample_sample, smooth_label

import itk
from itk import TubeTK as ttk

def convert_tre_to_mha(input_file: str, template_file: str, output_file: str):
    PixelType = itk.F
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]
        
    # Read tre file
    TubeFileReaderType = itk.SpatialObjectReader[Dimension]
        
    tubeFileReader = TubeFileReaderType.New()
    tubeFileReader.SetFileName(input_file)
    tubeFileReader.Update()

    tubes = tubeFileReader.GetGroup()


    # Read template image
    TemplateImageType = itk.Image[PixelType, Dimension]
    TemplateImageReaderType = itk.ImageFileReader[TemplateImageType]
        
    templateImageReader = TemplateImageReaderType.New()
    templateImageReader.SetFileName(template_file)
    templateImageReader.Update()

    templateImage = templateImageReader.GetOutput()
    TubesToImageFilterType = ttk.ConvertTubesToImage[TemplateImageType]
    tubesToImageFilter = TubesToImageFilterType.New()
    tubesToImageFilter.SetUseRadius(True)
    tubesToImageFilter.SetTemplateImage(templateImageReader.GetOutput())
    tubesToImageFilter.SetInput(tubes)
    tubesToImageFilter.Update()

    outputImage: itk.itkImagePython.itkImageF3 = tubesToImageFilter.GetOutput()

    # convert outputImage to mha file
    OutputImageWriterType = itk.ImageFileWriter[TemplateImageType]
    outputImageWriter = OutputImageWriterType.New()
    outputImageWriter.SetFileName(output_file)
    outputImageWriter.SetInput(outputImage)
    outputImageWriter.Update()


def convert_TubeTK(input_folder: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labelsTr"), exist_ok=True)

    for sample in os.listdir(input_folder):
        if "Normal" not in sample:
            continue
        sample_path = os.path.join(input_folder, sample)
        if "AuxillaryData" not in os.listdir(sample_path):
            continue
        
        print(f"Converting {sample}...")
        sample_num = sample.split("-")[1]

        image = sitk.ReadImage(os.path.join(sample_path, f"MRA/{sample.replace('-', '')}-MRA.mha"))
        convert_tre_to_mha(
            os.path.join(sample_path, f"AuxillaryData/VascularNetwork.tre"),
            os.path.join(sample_path, f"MRA/{sample.replace('-', '')}-MRA.mha"),
            os.path.join(sample_path, f"AuxillaryData/VascularNetwork.mha")
        )
        mask = sitk.ReadImage(os.path.join(sample_path, f"AuxillaryData/VascularNetwork.mha"))

        image, mask = resample_sample(image, mask, 2)
        # mask = smooth_label(mask)
        array, metadata = convert_sitk_image(image)
        print(f"Array type: {array.dtype}")
        metadata = metadata | calculate_metadata(array)
        save_array(array, os.path.join(output_dir, "imagesTr", sample_num))
        save_metadata(metadata, os.path.join(output_dir, "imagesTr", sample_num))

        array, metadata = convert_sitk_image(mask)
        array = array > 0
        save_array(array, os.path.join(output_dir, "labelsTr", sample_num))
        save_metadata(metadata, os.path.join(output_dir, "labelsTr", sample_num))
