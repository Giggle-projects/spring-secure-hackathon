package se.ton.t210.exception;

import lombok.Getter;
import org.springframework.http.HttpStatus;

@Getter
public class AuthException extends IllegalArgumentException {

    private final HttpStatus httpStatus;
    private final String message;

    public AuthException(HttpStatus httpStatus, String message) {
        this.httpStatus = httpStatus;
        this.message = message;
    }
}
