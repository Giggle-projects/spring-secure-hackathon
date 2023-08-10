package se.ton.t210.service;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import se.ton.t210.domain.MemberRepository;
import se.ton.t210.dto.LogInRequest;
import se.ton.t210.dto.SignUpRequest;
import se.ton.t210.exception.AuthException;
import se.ton.t210.redis.token.Token;
import se.ton.t210.redis.token.TokenRedisRepository;
import se.ton.t210.token.JwtUtils;
import se.ton.t210.token.TokenData;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletResponse;

@Transactional
@Service
public class MemberService {

    private final MemberRepository memberRepository;
    private final JwtUtils jwtUtils;
    private final TokenRedisRepository tokenRedisRepository;

    public MemberService(MemberRepository memberRepository, JwtUtils jwtUtils, TokenRedisRepository tokenRedisRepository) {
        this.memberRepository = memberRepository;
        this.jwtUtils = jwtUtils;
        this.tokenRedisRepository = tokenRedisRepository;
    }

    public void signUp(SignUpRequest request, HttpServletResponse response) {
        if (memberRepository.existsByUsername(request.getUsername())) {
            throw new AuthException(HttpStatus.CONFLICT, "username is already exists");
        }

        if (!request.isValidSignUp()) {
            throw new AuthException(HttpStatus.BAD_REQUEST, "password & re-password must be the same");
        }

        memberRepository.save(request.toEntity());

        TokenData token = jwtUtils.createTokenDataByUsername(request.getUsername());
        tokenRedisRepository.save(new Token(request.getUsername(), token.getAccessToken(), token.getRefreshToken()));

        saveTokenInCookie(response, token);
    }

    public void signIn(LogInRequest request, HttpServletResponse response) {
        if (!memberRepository.existsByUsernameAndPassword(request.getUsername(), request.getPassword())) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "The username or password is not valid.");
        }

        TokenData data = jwtUtils.createTokenDataByUsername(request.getUsername());
        tokenRedisRepository.save(new Token(request.getUsername(), data.getAccessToken(), data.getRefreshToken()));

        saveTokenInCookie(response, data);
    }

    public void reissueToken(TokenData data, HttpServletResponse response) {
        if (!jwtUtils.isExpired(data.getAccessToken())) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Reissue request is invalid");
        }

        String targetUsername = jwtUtils.getUsernameByToken(data.getRefreshToken());

        Token token = tokenRedisRepository.findById(targetUsername)
                .orElseThrow(() -> new AuthException(HttpStatus.UNAUTHORIZED, "Token is invalid"));

        if (!token.getRefreshToken().equals(data.getRefreshToken())) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Refresh Token is invalid");
        }

        TokenData reissueData = jwtUtils.createTokenDataByUsername(token.getUsername());
        tokenRedisRepository.save(new Token(token.getUsername(), reissueData.getAccessToken(), reissueData.getRefreshToken()));

        saveTokenInCookie(response, reissueData);
    }

    private void saveTokenInCookie(HttpServletResponse response, TokenData token) {
        Cookie accessTokenCookie = new Cookie("accessToken", token.getAccessToken());
        accessTokenCookie.setHttpOnly(true);
        accessTokenCookie.setPath("/");

        Cookie refreshTokenCookie = new Cookie("refreshToken", token.getRefreshToken());
        refreshTokenCookie.setHttpOnly(true);
        refreshTokenCookie.setPath("/");

        response.addCookie(accessTokenCookie);
        response.addCookie(refreshTokenCookie);
    }
}
