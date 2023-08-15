package se.ton.t210.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import se.ton.t210.cache.EmailAuthCodeCache;
import se.ton.t210.cache.EmailAuthCodeCacheRepository;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.MemberRepository;
import se.ton.t210.domain.TokenSecret;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.ApplicantCountResponse;
import se.ton.t210.dto.MemberTokens;
import se.ton.t210.dto.SignInRequest;
import se.ton.t210.dto.SignUpRequest;
import se.ton.t210.exception.AuthException;
import se.ton.t210.service.mail.MailServiceInterface;
import se.ton.t210.service.mail.form.SignUpAuthMailForm;
import se.ton.t210.service.token.TokenService;
import se.ton.t210.utils.auth.AuthCodeUtils;
import se.ton.t210.utils.http.CookieUtils;

import javax.servlet.http.HttpServletResponse;
import java.time.Duration;
import java.time.LocalTime;

@Transactional
@Service
public class MemberService {

    @Value("${auth.jwt.payload.key:email}")
    private String tokenKey;

    @Value("${auth.jwt.token.access.cookie.key:accessToken}")
    private String accessTokenCookieKey;

    @Value("${auth.jwt.token.refresh.cookie:refreshToken}")
    private String refreshTokenCookieKey;

    @Value("${auth.jwt.token.email.cookie.key:emailAuthToken}")
    private String emailAuthTokenCookieKey;

    @Value("${auth.mail.valid.time}")
    private Long mailValidTime;

    @Value("${auth.code.length}")
    private int authCodeLength;

    @Autowired
    TokenSecret tokenSecret;

    private final MemberRepository memberRepository;
    private final TokenService tokenService;
    private final EmailAuthCodeCacheRepository emailAuthCodeCacheRepository;
    private final MailServiceInterface mailServiceInterface;

    public MemberService(MemberRepository memberRepository, TokenService tokenService, EmailAuthCodeCacheRepository emailAuthCodeCacheRepository, MailServiceInterface mailServiceInterface) {
        this.memberRepository = memberRepository;
        this.tokenService = tokenService;
        this.emailAuthCodeCacheRepository = emailAuthCodeCacheRepository;
        this.mailServiceInterface = mailServiceInterface;
    }

    public void signUp(SignUpRequest request, String emailAuthToken, HttpServletResponse response) {
        if (memberRepository.existsByEmail(request.getEmail())) {
            throw new AuthException(HttpStatus.CONFLICT, "Email is already exists");
        }
        final String emailFromToken = tokenSecret.getPayloadValue(tokenKey, emailAuthToken);
        if (!request.getEmail().equals(emailFromToken)) {
            throw new AuthException(HttpStatus.FORBIDDEN, "It is different from the previous email information you entered.");
        }
        final Member member = memberRepository.save(request.toEntity());
        final MemberTokens tokens = tokenService.issue(member.getEmail());
        responseTokens(response, tokens);
    }

    public void signIn(SignInRequest request, HttpServletResponse response) {
        if (!memberRepository.existsByEmailAndPassword(request.getEmail(), request.getPassword())) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "The username or password is not valid.");
        }
        final MemberTokens tokens = tokenService.issue(request.getEmail());
        responseTokens(response, tokens);
    }

    public void reissueToken(String accessToken, String refreshToken, HttpServletResponse response) {
        final MemberTokens tokens = tokenService.reissue(accessToken, refreshToken);
        responseTokens(response, tokens);
    }

    public void reissuePwd(String email, String newPwd) {
        Member member = memberRepository.findByEmail(email).orElseThrow(() ->
                new AuthException(HttpStatus.NOT_FOUND, "User is not found"));
        if (member.getEmail().equals(newPwd)) {
            throw new IllegalArgumentException("The password you want to change must be different from the previous password.");
        }
        member.reissuePwd(newPwd);
        memberRepository.save(member);
    }

    public void validateEmailAuthCode(String email, String authCode) {
        EmailAuthCodeCache emailAuthCodeCache = emailAuthCodeCacheRepository.findById(email).orElseThrow(() ->
                new AuthException(HttpStatus.NOT_FOUND, "Email not found"));
        long afterSeconds = Duration.between(emailAuthCodeCache.getCreatedTime(), LocalTime.now()).getSeconds();
        if (afterSeconds > mailValidTime) {
            throw new AuthException(HttpStatus.REQUEST_TIMEOUT, "Email valid time is exceed");
        }
        if (!emailAuthCodeCache.getAuthCode().equals(authCode)) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "AuthCode is not correct");
        }
        if (!emailAuthCodeCache.getEmail().equals(email)) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "email is not correct");
        }
    }

    public void issueEmailToken(HttpServletResponse response, String email) {
        final String emailAuthToken = tokenService.issueMailToken(email);
        CookieUtils.loadHttpOnlyCookie(response, emailAuthTokenCookieKey, emailAuthToken);
    }

    public void issueToken(HttpServletResponse response, String email) {
        final MemberTokens tokens = tokenService.issue(email);
        responseTokens(response, tokens);
    }

    private void responseTokens(HttpServletResponse response, MemberTokens tokens) {
        CookieUtils.loadHttpOnlyCookie(response, accessTokenCookieKey, tokens.getAccessToken());
        CookieUtils.loadHttpOnlyCookie(response, refreshTokenCookieKey, tokens.getRefreshToken());
    }

    public void sendEmailAuthMail(String userEmailAddress) {
        String emailAuthCode = AuthCodeUtils.generate(authCodeLength);
        mailServiceInterface.sendMail(userEmailAddress, new SignUpAuthMailForm(emailAuthCode));
        emailAuthCodeCacheRepository.save(new EmailAuthCodeCache(userEmailAddress, emailAuthCode, LocalTime.now()));
    }

    public ApplicantCountResponse countApplicant(ApplicationType applicationType) {
        final int count = memberRepository.countByApplicationType(applicationType);
        return new ApplicantCountResponse(count);
    }

    public void isExistEmail(String email) {
        if (!memberRepository.existsByEmail(email)) {
            throw new AuthException(HttpStatus.CONFLICT, "Email is not exists");
        }
    }

    public void isNotExistEmail(String email) {
        if (memberRepository.existsByEmail(email)) {
            throw new AuthException(HttpStatus.CONFLICT, "Email is already exists");
        }
    }
}
