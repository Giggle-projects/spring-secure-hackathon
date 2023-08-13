package se.ton.t210.domain.type;

public enum ApplicationType {
    PoliceOfficerMale("/static/default.jpg", "경찰직공무원(남)"),
    PoliceOfficerFemale("/static/default.jpg", "경찰직공무원(여)"),
    FireOfficerMale("/static/default.jpg", "소방직공무원(남)"),
    FireOfficerFemale("/static/default.jpg", "소방직공무원(여)"),
    CorrectionalOfficerMale("/static/default.jpg", "교정직공무원(남)"),
    CorrectionalOfficerFemale("/static/default.jpg", "교정직공무원(여)");

    private final String iconImageUrl;
    private final String name;

    ApplicationType(String iconImageUrl, String name) {
        this.iconImageUrl = iconImageUrl;
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public String iconImageUrl() {
        return iconImageUrl;
    }
}
