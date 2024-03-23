import os
import xml.etree.ElementTree as ET

def extract_labels(file_name, output_file):
    print("Currently on:", file_name)
    with open(file_name, 'r') as file_in, open(output_file, 'w') as file_out:
        for line in file_in:
            parts = line.strip().split()
            print(parts)

            start_frame, end_frame = map(int, parts[0].split('-'))
            label = parts[1]

            if start_frame == 1: # For the 1st frame
                file_out.write(f"{label}\n")

            for frame in range(start_frame, end_frame):
                file_out.write(f"{label}\n")
    print(f"Labels extracted and saved to '{output_file}'")

# Function to extract labels from a .xgtf file and save them to a new file
def extract_labels_xgtf(file_name, output_file):
    tree = ET.parse(file_name)
    root = tree.getroot()

    # Extract total frames
    num_frames_element = root.find(".//{http://lamp.cfar.umd.edu/viper#}attribute[@name='NUMFRAMES']/{http://lamp.cfar.umd.edu/viperdata#}dvalue")
    if num_frames_element is not None:
        total_frames = int(num_frames_element.get("value"))
    else:
        print("Error: Unable to extract total frames from the .xgtf file.")
        return

    with open(output_file, 'w') as file_out:
        # Create a dictionary to store labels for each frame
        frame_labels = {}

        # Extract labels from object tags
        for object_tag in root.findall(".//{http://lamp.cfar.umd.edu/viper#}object"):
            framespan = object_tag.get("framespan")
            label = object_tag.get("name").replace(" ", "_")  # Replace spaces with underscores
            if framespan and label:
                frames = framespan.split()
                for frame_range in frames:
                    if ":" in frame_range:
                        start_frame, end_frame = map(int, frame_range.split(":"))
                        for frame in range(start_frame, end_frame + 1):
                            frame_labels[frame] = label

        # Write labels to the output file, matching line numbers with frame numbers
        for frame_num in range(1, total_frames + 1):
            label = frame_labels.get(frame_num, "bg")
            file_out.write(f"{label}\n")

    print(f"Labels extracted and saved to '{output_file}'")

if __name__ == '__main__':

  # Read label information from the file
  file_name = 'D:/FYP/Datasets/breakfast/BreakfastII_15fps_qvga_sync/P34/cam01/P34_coffee.avi.labels'
  _, extension = os.path.splitext(file_name)

  # Save labels to a file without an extension
  output_file = 'extracted_labels'
  
  if extension == '.xgtf':
    extract_labels_xgtf(file_name, output_file)
  else:
    extract_labels(file_name, output_file)