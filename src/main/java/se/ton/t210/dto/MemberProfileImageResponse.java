package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class MemberProfileImageResponse {
    
    private final String url;

    public MemberProfileImageResponse(String url) {
        this.url = url;
    }
}
