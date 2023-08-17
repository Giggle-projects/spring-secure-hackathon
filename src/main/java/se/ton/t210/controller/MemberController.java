package se.ton.t210.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import se.ton.t210.configuration.annotation.LoginMember;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.*;
import se.ton.t210.service.MemberService;
import se.ton.t210.service.image.S3Service;

import javax.servlet.http.HttpServletResponse;
import javax.validation.Valid;
import javax.validation.constraints.Email;

@RestController
public class MemberController {

    private final MemberService memberService;
    private final S3Service s3Service;

    public MemberController(MemberService memberService, S3Service s3Service) {
        this.memberService = memberService;
        this.s3Service = s3Service;
    }

    @PostMapping("/api/member/signUp")
    public ResponseEntity<Void> signUp(@RequestBody @Valid SignUpRequest request,
                                       @CookieValue String emailAuthToken,
                                       HttpServletResponse response) {
        memberService.signUp(request, emailAuthToken, response);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/api/member/signIn")
    public ResponseEntity<Void> signIn(@RequestBody @Valid SignInRequest request,
                                       HttpServletResponse response) {
        memberService.signIn(request, response);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/api/reissue/token")
    public ResponseEntity<Void> reissueToken(@CookieValue String accessToken,
                                             @CookieValue String refreshToken,
                                             HttpServletResponse response) {
        memberService.reissueToken(accessToken, refreshToken, response);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/api/remove/token")
    public ResponseEntity<Void> removeCookie(HttpServletResponse response,
                                             String cookieKey) {
        memberService.removeToken(response, cookieKey);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/api/send/mail")
    public ResponseEntity<Void> sendEmailAuthMail(@Valid @Email String email) {
        memberService.sendEmailAuthMail(email);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/api/reset/userInfo")
    public ResponseEntity<Void> resetUserInfo(@LoginMember LoginMemberInfo member,
                                              @RequestBody @Valid ResetPersonalInfoRequest request) {
        memberService.resetUserInfo(member, request);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/api/signUp/valid/authCode")
    public ResponseEntity<Void> validateAuthCodeFromSignUp(@RequestBody ValidateAuthCodeRequest request,
                                                           HttpServletResponse response) {
        memberService.validateEmailAuthCode(request.getEmail(), request.getAuthCode());
        memberService.issueEmailToken(response, request.getEmail());
        return ResponseEntity.ok().build();
    }

    @PostMapping("/api/forgetPwd/valid/authCode")
    public ResponseEntity<Void> validateAuthCodeFromForgetPwd(@RequestBody ValidateAuthCodeRequest request,
                                                              HttpServletResponse response) {
        memberService.validateEmailAuthCode(request.getEmail(), request.getAuthCode());
        memberService.issueToken(response, request.getEmail());
        return ResponseEntity.ok().build();
    }

    @GetMapping("/api/notExist/email")
    public ResponseEntity<Void> validNotExistElement(String email) {
        memberService.isNotExistEmail(email);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/api/isExist/email")
    public ResponseEntity<Void> validExistElement(String email) {
        memberService.isExistEmail(email);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/api/applicant/count")
    public ResponseEntity<ApplicantCountResponse> applicantCount() {
        final Member member = new Member(1L, "name", "email", "password", ApplicationType.PoliceOfficerMale);
        final ApplicantCountResponse applicantCountResponse = memberService.countApplicant(member.getApplicationType());
        return ResponseEntity.ok(applicantCountResponse);
    }

    @GetMapping("/api/member/personal/info")
    public ResponseEntity<MemberPersonalInfoResponse> personalInfo(@CookieValue String accessToken) {
        final MemberPersonalInfoResponse response = memberService.getPersonalInfo(accessToken);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/api/member/me")
    public ResponseEntity<MemberResponse> me(@LoginMember LoginMemberInfo loginMember) {
        final MemberResponse response = MemberResponse.of(loginMember);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/api/member/profile/image")
    public ResponseEntity<MemberProfileImageResponse> memberProfileImage(@LoginMember LoginMemberInfo memberInfo) {
        final String memberProfileUrl = memberService.getMemberProfileImage(memberInfo);
        final MemberProfileImageResponse response = new MemberProfileImageResponse(memberProfileUrl);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/upload/image")
    public ResponseEntity<Void> post(@LoginMember LoginMemberInfo memberInfo,
                                     @RequestPart("file") MultipartFile multipartFile) {
        s3Service.saveMemberProfileImage(memberInfo, multipartFile);
        return ResponseEntity.ok().build();
    }
}
