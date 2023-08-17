package se.ton.t210.controller;

import com.amazonaws.services.s3.AmazonS3Client;
import com.amazonaws.services.s3.model.ObjectMetadata;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import se.ton.t210.configuration.annotation.LoginMember;
import se.ton.t210.dto.LoginMemberInfo;
import se.ton.t210.service.MemberService;

@RestController
public class UploadController {

    private final AmazonS3Client amazonS3Client;
    private final String bucket = "nick-terraform-test-bucket";
    private final MemberService memberService;

    @Autowired
    public UploadController(AmazonS3Client amazonS3Client, MemberService memberService) {
        this.amazonS3Client = amazonS3Client;
        this.memberService = memberService;
    }

    @PostMapping("/api/image/upload")
    public ResponseEntity<String> post(@LoginMember LoginMemberInfo loginInfo,
                                       @RequestPart("file") MultipartFile multipartFile) {
        final String originalFilename = multipartFile.getOriginalFilename();
        ObjectMetadata metadata = new ObjectMetadata();
        metadata.setContentLength(multipartFile.getSize());
        metadata.setContentType(multipartFile.getContentType());
        try {
            amazonS3Client.putObject(bucket, originalFilename, multipartFile.getInputStream(), metadata);
            String image = amazonS3Client.getUrl(bucket, originalFilename).toString();
            memberService.uploadProfileImage(loginInfo, image);
            return ResponseEntity.ok(image);
        } catch (Exception e) {
            e.printStackTrace();
            throw new IllegalArgumentException();
        }
    }
}
