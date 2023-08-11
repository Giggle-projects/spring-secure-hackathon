package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class Email {

    private final String title;

    private final String content;

    private final String toAddress;

    public Email(String title, String content, String toAddress) {
        this.title = title;
        this.content = content;
        this.toAddress = toAddress;
    }
}
