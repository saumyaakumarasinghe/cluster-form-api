import base64
from io import BytesIO
from PIL import Image, ImageEnhance


class ImageService:
    def compressed_image_from_base64(
        self, base64_string, max_size=(400, 400), target_size_kb=100
    ):
        try:
            # Check if the base64 string starts with the data URL prefix, and remove it if present
            if base64_string.startswith("data:image"):
                base64_string = base64_string.split(",")[1]  # Remove prefix

            # Decode the base64 string into image data
            image_data = base64.b64decode(base64_string)
            original_size_kb = len(image_data) / 1024

            # Create a BytesIO stream from the decoded image data
            input_buffer = BytesIO(image_data)
            image = Image.open(input_buffer)

            # Convert to RGB if needed (for formats with transparency)
            if image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            ):
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(
                    image, mask=image.split()[3] if image.mode == "RGBA" else None
                )
                image = background

            # Resize if needed - use progressive sizing approach
            original_width, original_height = image.size
            if original_width > max_size[0] or original_height > max_size[1]:
                image.thumbnail(max_size, Image.LANCZOS)

            # Enhance sharpness slightly to compensate for any resize blur
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)

            # Try different formats and qualities to find the best balance
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

                # If we're already under target size, we can stop
                if test_size_kb <= target_size_kb:
                    break

            # Test WEBP format as an alternative
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

                    # If we're already under target size, we can stop
                    if test_size_kb <= target_size_kb:
                        break
            except Exception:
                # WEBP might not be supported, continue with JPEG
                pass

            # Re-encode the image to base64
            compressed_base64 = base64.b64encode(best_data).decode("utf-8")

            # MIME type mapping
            mime_type = "jpeg" if best_format == "JPEG" else "webp"

            # Return the full base64 string with the correct format
            full_base64 = f"data:image/{mime_type};base64,{compressed_base64}"

            # Create a shortened preview for logging
            shortened_base64 = f"{full_base64[:50]}..."

            # Log compression statistics
            compression_ratio = original_size_kb / best_size if best_size > 0 else 0
            print(
                f"Image optimized: {image.width}x{image.height}, Format: {best_format}, "
                f"Quality: {best_quality}, Size: {best_size:.1f}KB ({compression_ratio:.1f}x smaller)"
            )

            return shortened_base64, full_base64

        except Exception as e:
            print(f"Error processing image: {e}")
            return None, None
