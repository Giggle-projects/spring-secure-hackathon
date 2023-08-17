package se.ton.t210.controller;

import static org.springframework.http.MediaType.APPLICATION_JSON_VALUE;

import com.amazonaws.AmazonServiceException;
import com.amazonaws.services.s3.AmazonS3Client;
import com.amazonaws.services.s3.model.CannedAccessControlList;
import com.amazonaws.services.s3.model.ObjectMetadata;
import com.amazonaws.services.s3.model.PutObjectRequest;
import com.amazonaws.services.s3.model.S3Object;
import com.amazonaws.services.s3.model.S3ObjectInputStream;
import com.amazonaws.util.IOUtils;
import java.io.IOException;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequiredArgsConstructor
public class UploadController {

  private final AmazonS3Client amazonS3Client;
  private final String bucket = "nick-terraform-test-bucket";

  @PostMapping("/test")
  public ResponseEntity<String> post(@RequestPart("file") MultipartFile multipartFile) {
    String originalFilename = multipartFile.getOriginalFilename();

    ObjectMetadata metadata = new ObjectMetadata();
    metadata.setContentLength(multipartFile.getSize());
    metadata.setContentType(multipartFile.getContentType());

    try {
      amazonS3Client.putObject(bucket, "test/"+originalFilename, multipartFile.getInputStream(), metadata);
      return ResponseEntity.ok(amazonS3Client.getUrl(bucket, originalFilename).toString());
    } catch (Exception e) {
      e.printStackTrace();
      throw new IllegalArgumentException();
    }
  }
}
