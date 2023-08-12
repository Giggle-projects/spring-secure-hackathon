package se.ton.t210.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import se.ton.t210.dto.*;
import se.ton.t210.service.MailAuthService;
import se.ton.t210.service.MemberService;

import javax.servlet.http.HttpServletResponse;
import javax.validation.Valid;

@RequestMapping("/api/auth")
@RestController
public class MemberController {

    private final MemberService memberService;
    private final MailAuthService mailAuthService;

    public MemberController(MemberService memberService, MailAuthService mailAuthService) {
        this.memberService = memberService;
        this.mailAuthService = mailAuthService;
    }

    @PostMapping("/signUp")
    public ResponseEntity<Void> signUp(@RequestBody @Valid SignUpRequest request,
                                       HttpServletResponse response) {
        request.validateSignUpRequest();
        String emailByToken = "devygwan@gmail.com";  //emailAuthToken에서 추출한 email 값(임시)
        memberService.signUp(request, emailByToken, response);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/signIn")
    public ResponseEntity<Void> signIn(@RequestBody @Valid SignInRequest request,
                                       HttpServletResponse response) {
        memberService.signIn(request, response);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/reissue/token")
    public ResponseEntity<Void> reissueToken(@CookieValue String accessToken,
                                             @CookieValue String refreshToken,
                                             HttpServletResponse response) {
        memberService.reissueToken(accessToken, refreshToken, response);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/send/mail")
    public ResponseEntity<Void> sendEmailAuthMail(@RequestBody EmailRequest request) {
        mailAuthService.sendEmailAuthMail(request.getEmail());
        return ResponseEntity.ok().build();
    }

    @PostMapping("/check/authCode")
    public ResponseEntity<Void> validateAuthCode(@RequestBody ValidateAuthCodeRequest request,
                                                 HttpServletResponse response) {
        mailAuthService.validateAuthCode(request.getEmail(), request.getAuthCode(), response);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/reissue/pwd")
    public ResponseEntity<Void> reissuePwd(@RequestBody ReissuePwdRequest request) {
        String email = "devygwan@gmail.com"; //임시
        memberService.reissuePwd(email, request.getPassword());
        return ResponseEntity.ok().build();
    }
}
