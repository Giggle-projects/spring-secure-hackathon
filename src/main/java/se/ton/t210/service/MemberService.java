package se.ton.t210.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import se.ton.t210.cache.EmailAuthCodeCache;
import se.ton.t210.cache.EmailAuthCodeCacheRepository;
import se.ton.t210.domain.*;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.*;
import se.ton.t210.exception.AuthException;
import se.ton.t210.service.mail.MailServiceInterface;
import se.ton.t210.service.mail.form.SignUpAuthMailForm;
import se.ton.t210.service.token.TokenService;
import se.ton.t210.utils.auth.RandomCodeUtils;
import se.ton.t210.utils.http.CookieUtils;

import javax.servlet.http.HttpServletResponse;
import java.time.LocalTime;
import java.util.Objects;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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

    private final TokenSecret tokenSecret;
    private final MemberRepository memberRepository;
    private final PasswordSaltRepository saltRepository;
    private final TokenService tokenService;
    private final EmailAuthCodeCacheRepository emailAuthCodeCacheRepository;
    private final MailServiceInterface mailServiceInterface;
    private final MemberProfileRepository memberProfileRepository;

    public MemberService(TokenSecret tokenSecret, MemberRepository memberRepository,
                         PasswordSaltRepository saltRepository, TokenService tokenService,
                         EmailAuthCodeCacheRepository emailAuthCodeCacheRepository,
                         MailServiceInterface mailServiceInterface, MemberProfileRepository memberProfileRepository) {
        this.tokenSecret = tokenSecret;
        this.memberRepository = memberRepository;
        this.saltRepository = saltRepository;
        this.tokenService = tokenService;
        this.emailAuthCodeCacheRepository = emailAuthCodeCacheRepository;
        this.mailServiceInterface = mailServiceInterface;
        this.memberProfileRepository = memberProfileRepository;
    }

    public void signUp(SignUpRequest request, String emailAuthToken, HttpServletResponse response) {
        if (memberRepository.existsByEmail(request.getEmail())) {
            throw new AuthException(HttpStatus.CONFLICT, "Email is already exists");
        }
        final String emailFromToken = tokenSecret.getPayloadValue(tokenKey, emailAuthToken);
        if (!request.getEmail().equals(emailFromToken)) {
            throw new AuthException(HttpStatus.FORBIDDEN, "It is different from the previous email information you entered.");
        }

        final EncryptPassword encryptPassword = EncryptPassword.encryptFrom(request.getPassword());
        final Member member = request.toEntity()
                .updatePasswordWith(encryptPassword.getEncrypted());
        memberRepository.save(member);
        saltRepository.save(new PasswordSalt(member.getId(), encryptPassword.getSalt()));

        final MemberTokens tokens = tokenService.issue(member.getEmail());
        responseTokens(response, tokens);
    }

    public void signIn(SignInRequest request, HttpServletResponse response) {
        final Member member = memberRepository.findByEmail(request.getEmail()).orElseThrow();
        final PasswordSalt salt = saltRepository.findByMemberId(member.getId()).orElseThrow();
        member.validatePassword(request.getPassword(), salt);
        final MemberTokens tokens = tokenService.issue(member.getEmail());
        responseTokens(response, tokens);
    }

    public void reissueToken(String accessToken, String refreshToken, HttpServletResponse response) {
        final MemberTokens tokens = tokenService.reissue(accessToken, refreshToken);
        responseTokens(response, tokens);
    }

    public void reissueToken(String refreshToken, HttpServletResponse response) {
        final MemberTokens tokens = tokenService.issue(refreshToken);
        responseTokens(response, tokens);
    }

    public void validateEmailAuthCode(String email, String authCode) {
        final EmailAuthCodeCache emailAuthCodeCache = emailAuthCodeCacheRepository.findById(email).orElseThrow(() ->
                new AuthException(HttpStatus.NOT_FOUND, "Email not found")
        );
        emailAuthCodeCache.checkValidTime(mailValidTime);
        emailAuthCodeCache.checkAuthCodeSame(authCode);
        emailAuthCodeCache.checkEmailSame(email);
    }

    public void issueEmailToken(HttpServletResponse response, String email) {
        final String emailAuthToken = tokenService.issueMailToken(email);
        CookieUtils.loadHttpOnlyCookie(response, emailAuthTokenCookieKey, emailAuthToken);
    }

    public void removeToken(HttpServletResponse response, String cookieKey) {
        CookieUtils.removeHttpOnlyCookie(response, cookieKey);
    }

    public void issueToken(HttpServletResponse response, String email) {
        final MemberTokens tokens = tokenService.issue(email);
        responseTokens(response, tokens);
    }

    private void responseTokens(HttpServletResponse response, MemberTokens tokens) {
        CookieUtils.loadHttpOnlyCookie(response, accessTokenCookieKey, tokens.getAccessToken());
        CookieUtils.loadHttpOnlyCookie(response, refreshTokenCookieKey, tokens.getRefreshToken());
    }

    public void sendEmailAuthMail(String email) {
        final String emailAuthCode = RandomCodeUtils.generate();
        mailServiceInterface.sendMail(email, new SignUpAuthMailForm(emailAuthCode));
        emailAuthCodeCacheRepository.save(new EmailAuthCodeCache(email, emailAuthCode, LocalTime.now()));
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

    public MemberPersonalInfoResponse getPersonalInfo(String accessToken) {
        final String email = tokenSecret.getPayloadValue(tokenKey, accessToken);
        return memberRepository.getMemberByEmail(email);
    }

    public void resetUserInfo(LoginMemberInfo memberInfo, ResetPersonalInfoRequest request) {
        Member member = memberRepository.findById(memberInfo.getId()).orElseThrow(() -> {
            throw new AuthException(HttpStatus.CONFLICT, "Email is not exists");
        });
        request.getApplicationType().ifPresent(member::resetApplicationType);
        request.getPassword().ifPresent(password -> reissuePwd(member, password));
    }

    private void reissuePwd(Member oldMember, String newPwd) {
        if(Objects.equals(newPwd, "")) {
            return;
        }
        if (oldMember.getEmail().equals(newPwd)) {
            throw new IllegalArgumentException("Password can't be same with email");
        }
        String passwordPattern = "(?=.*[0-9])(?=.*[a-zA-Z])(?=.*\\W)(?=\\S+$).{9,16}";
        Pattern pattern = Pattern.compile(passwordPattern);
        Matcher matcher = pattern.matcher(newPwd);
        if (!matcher.matches()) {
            throw new IllegalArgumentException("비밀번호 형식이 올바르지 않습니다.");
        }
        final EncryptPassword encryptPassword = EncryptPassword.encryptFrom(newPwd);
        final Member member = oldMember.updatePasswordWith(encryptPassword.getEncrypted());
        saltRepository.save(new PasswordSalt(member.getId(), encryptPassword.getSalt()));
    }

    public String getMemberProfileImage(LoginMemberInfo memberInfo) {
        final Long memberId = memberInfo.getId();
        Optional<MemberProfileImage> memberProfileImage = memberProfileRepository.findByMemberId(memberId);
        if (memberProfileImage.isPresent()) {
            return memberProfileImage.get().getImageUrl();
        }
        return memberInfo.getApplicationType().iconImageUrl();
    }

    public void uploadProfileImage(LoginMemberInfo memberInfo, String imageUrl) {
        memberProfileRepository.save(new MemberProfileImage(memberInfo.getId(), imageUrl));
    }
}

