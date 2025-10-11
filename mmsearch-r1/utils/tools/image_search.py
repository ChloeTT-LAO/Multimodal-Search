from PIL import Image
import numpy as np


def call_image_search(image_url: str):
    """
    Placeholder function for an image-based search tool.

    This function simulates image search behavior based on a provided image URL.
    The actual search logic (e.g., calling a search engine or internal API)
    has been omitted due to privacy or dependency restrictions.

    Args:
        image_url (str): A URL or internal key representing the query image.

    Returns:
        tool_returned_str (str): A placeholder string summarizing fake image search results.
        tool_returned_images (List[PIL.Image.Image]): A list of dummy PIL image objects
            representing search result thumbnails.
        tool_stat (dict): A dictionary indicating tool status and optional metadata.
    """

    print(
        "[Warning] You are currently using a *fake* implementation of the image search tool.\n"
        "This placeholder is intended for testing and does not perform real image retrieval.\n"
        "To enable this feature, please replace the function with logic that calls your own image search system or API."
    )

    # Init
    tool_returned_images = []
    tool_returned_str = "[Image Search Results] The result of the image search consists of web page information related to the image from the user's original question. Each result includes the main image from the web page and its title, ranked in descending order of search relevance, as demonstrated below:\n"

    # Simulated 3 searched results
    for i in range(3):
        # Create a dummy RGB image (e.g., 64x64 solid color)
        dummy_img = Image.fromarray(np.full((64, 64, 3), fill_value=100 + i * 30, dtype=np.uint8))
        tool_returned_images.append(dummy_img)
        # Simulate search result with image marker and fake title
        tool_returned_str += f"{i+1}. image: <|vision_start|><|image_pad|><|vision_end|>\ntitle: example webpage title {i+1}\n"

    # Simulated tool status
    tool_success = True
    if not tool_success:
        tool_returned_str = "[Image Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
        tool_returned_images = []
    tool_stat = {
        "success": tool_success,
        "num_images": len(tool_returned_images),
    }

    return tool_returned_str, tool_returned_images, tool_stat