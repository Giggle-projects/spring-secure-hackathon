package se.ton.t210.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.*;
import se.ton.t210.service.MemberService;

import javax.servlet.http.HttpServletResponse;
import javax.validation.Valid;

@RestController
public class MemberController {

    private final MemberService memberService;

    public MemberController(MemberService memberService) {
        this.memberService = memberService;
    }

    @PostMapping("/api/member/signUp")
    public ResponseEntity<Void> signUp(@RequestBody @Valid SignUpRequest request,
                                       @CookieValue String emailAuthToken,
                                       HttpServletResponse response) {
        System.out.println(emailAuthToken);
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

    @GetMapping("/api/send/mail")
    public ResponseEntity<Void> sendEmailAuthMail(String email) {
        memberService.sendEmailAuthMail(email);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/api/valid/authCode")
    public ResponseEntity<Void> validateAuthCode(@RequestBody ValidateAuthCodeRequest request,
                                                 HttpServletResponse response) {
        memberService.validateEmailAuthCode(request.getEmail(), request.getAuthCode());
        memberService.issueEmailToken(request.getEmail(), response);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/api/reissue/pwd")
    public ResponseEntity<Void> reissuePwd(@RequestBody ReissuePwdRequest request) {
        String email = "devygwan@gmail.com"; //임시
        memberService.reissuePwd(email, request.getPassword());
        return ResponseEntity.ok().build();
    }

    @GetMapping("/api/applicant/count")
    public ResponseEntity<ApplicantCountResponse> applicantCount() {
        final Member member = new Member(1L, "name", "email", "password", ApplicationType.PoliceOfficerMale);
        final ApplicantCountResponse applicantCountResponse = memberService.countApplicant(member.getApplicationType());
        return ResponseEntity.ok(applicantCountResponse);
    }

    @GetMapping("/api/member/me")
    public ResponseEntity<MemberResponse> me() {
        final Member member = new Member(1L, "name", "email", "password", ApplicationType.PoliceOfficerMale);
        final MemberResponse response = MemberResponse.of(member);
        return ResponseEntity.ok(response);
    }
}
