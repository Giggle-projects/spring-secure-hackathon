package se.ton.t210.service.image;

import com.amazonaws.services.s3.AmazonS3Client;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import se.ton.t210.domain.BlackListRepository;
import se.ton.t210.dto.LoginMemberInfo;
import se.ton.t210.exception.AuthException;
import se.ton.t210.service.MemberService;
import se.ton.t210.utils.s3.CreateImageUtils;

@Service
public class S3Service {

    @Value("${s3.amazon.bucket}")
    private String bucket;

    private final BlackListRepository blackListRepository;
    private final AmazonS3Client amazonS3Client;
    private final MemberService memberService;

    public S3Service(BlackListRepository blackListRepository, AmazonS3Client amazonS3Client, MemberService memberService) {
        this.blackListRepository = blackListRepository;
        this.amazonS3Client = amazonS3Client;
        this.memberService = memberService;
    }

    public void saveMemberProfileImage(LoginMemberInfo member, MultipartFile multipartFile) {
        if (blackListRepository.existsBlackListByMemberId(member.getId())) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Unauthorized");
        }
        final String image = CreateImageUtils.createImage(multipartFile, amazonS3Client, bucket);
        memberService.uploadProfileImage(member, image);
    }
}
