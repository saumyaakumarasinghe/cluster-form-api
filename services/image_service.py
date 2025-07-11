"""
Image Service for processing and compressing visualization images.
Handles base64 encoding/decoding, image compression, and format optimization.
"""

import base64
from io import BytesIO
from PIL import Image, ImageEnhance


class ImageService:
    """
    Service for processing and optimizing images for web display.
    Handles base64 encoding, compression, and format conversion.
    """

    def compressed_image_from_base64(
        self, base64_string, max_size=(300, 300), target_size_kb=100
    ):
        """
        Compress and optimize a base64-encoded image for web display.

        Args:
            base64_string (str): Base64-encoded image string
            max_size (tuple): Maximum dimensions (width, height) in pixels
            target_size_kb (int): Target file size in kilobytes

        Returns:
            tuple: (shortened_preview, full_base64_string) or (None, None) if failed
        """
        try:
            # Remove data URL prefix if present
            if base64_string.startswith("data:image"):
                base64_string = base64_string.split(",")[1]

            # Decode base64 to image data
            image_data = base64.b64decode(base64_string)
            original_size_kb = len(image_data) / 1024

            # Create image from bytes
            input_buffer = BytesIO(image_data)
            image = Image.open(input_buffer)

            # Convert to RGB if needed (handle transparency)
            if image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            ):
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(
                    image, mask=image.split()[3] if image.mode == "RGBA" else None
                )
                image = background

            # Resize if needed
            original_width, original_height = image.size
            if original_width > max_size[0] or original_height > max_size[1]:
                image.thumbnail(max_size, Image.LANCZOS)

            # Enhance sharpness slightly
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)

            # Try different formats and qualities for optimal compression
            best_data = None
            best_size = float("inf")
            best_format = "JPEG"
            best_quality = 85

            # Test JPEG format with different qualities
            for quality in [85, 75, 65]:
                output_buffer = BytesIO()
                image.save(output_buffer, format="JPEG", quality=quality, optimize=True)
                output_buffer.seek(0)
                test_data = output_buffer.getvalue()
                test_size_kb = len(test_data) / 1024
                if test_size_kb < best_size:
                    best_data = test_data
                    best_size = test_size_kb
                    best_quality = quality
                    best_format = "JPEG"
                if test_size_kb <= target_size_kb:
                    break

            # Test WEBP format as alternative (better compression)
            try:
                for quality in [85, 75, 65]:
                    output_buffer = BytesIO()
                    image.save(output_buffer, format="WEBP", quality=quality)
                    output_buffer.seek(0)
                    test_data = output_buffer.getvalue()
                    test_size_kb = len(test_data) / 1024
                    if test_size_kb < best_size:
                        best_data = test_data
                        best_size = test_size_kb
                        best_quality = quality
                        best_format = "WEBP"
                    if test_size_kb <= target_size_kb:
                        break
            except Exception:
                pass  # WEBP might not be supported

            # Encode to base64
            compressed_base64 = base64.b64encode(best_data).decode("utf-8")
            mime_type = "jpeg" if best_format == "JPEG" else "webp"
            full_base64 = f"data:image/{mime_type};base64,{compressed_base64}"
            shortened_base64 = f"{full_base64[:50]}..."

            # Log compression stats
            compression_ratio = original_size_kb / best_size if best_size > 0 else 0
            print(
                f"Image optimized: {image.width}x{image.height}, Format: {best_format}, "
                f"Quality: {best_quality}, Size: {best_size:.1f}KB ({compression_ratio:.1f}x smaller)"
            )

            return shortened_base64, full_base64

        except Exception as e:
            print(f"Error processing image: {e}")
            return None, None
