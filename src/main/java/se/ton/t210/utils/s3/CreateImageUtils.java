package se.ton.t210.utils.s3;

import com.amazonaws.services.s3.AmazonS3Client;
import com.amazonaws.services.s3.model.ObjectMetadata;
import org.springframework.web.multipart.MultipartFile;

public class CreateImageUtils {

    public static String createImage(MultipartFile multipartFile, AmazonS3Client amazonS3Client, String bucket) {
        final String originalFilename = multipartFile.getOriginalFilename();
        ObjectMetadata metadata = new ObjectMetadata();
        metadata.setContentLength(multipartFile.getSize());
        metadata.setContentType(multipartFile.getContentType());
        try {
            amazonS3Client.putObject(bucket, originalFilename, multipartFile.getInputStream(), metadata);
            return amazonS3Client.getUrl(bucket, originalFilename).toString();
        } catch (Exception e) {
            e.printStackTrace();
            throw new IllegalArgumentException();
        }
    }
}
