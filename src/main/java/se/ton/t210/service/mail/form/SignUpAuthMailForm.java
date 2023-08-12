package se.ton.t210.service.mail.form;

public class SignUpAuthMailForm implements MailForm {

    private final String authCode;

    public SignUpAuthMailForm(String authCode) {
        this.authCode = authCode;
    }

    @Override
    public String title() {
        return "CSPFT Email Authentication";
    }

    @Override
    public String body() {
        return "Email Authentication code is : " + authCode;
    }
}
