package se.ton.t210.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import se.ton.t210.dto.*;
import se.ton.t210.service.AuthService;
import se.ton.t210.service.MemberService;

import javax.servlet.http.HttpServletResponse;
import javax.validation.Valid;

@RequestMapping("/api/auth")
@RestController
public class MemberController {

    private final MemberService memberService;
    private final AuthService authService;

    public MemberController(MemberService memberService, AuthService authService) {
        this.memberService = memberService;
        this.authService = authService;
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
        authService.sendEmailAuthMail(request.getEmail());
        return ResponseEntity.ok().build();
    }

    @PostMapping("/check/authCode")
    public ResponseEntity<Void> validateAuthCode(@RequestBody ValidateAuthCodeRequest request,
                                                 HttpServletResponse response) {
        authService.validateAuthCode(request.getEmail(), request.getAuthCode(), response);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/reissue/pwd")
    public ResponseEntity<Void> reissuePwd(@RequestBody ReissuePwdRequest request) {
        String email = "devygwan@gmail.com"; //임시
        memberService.reissuePwd(email, request.getPassword());
        return ResponseEntity.ok().build();
    }
}
