package se.ton.t210.domain;

public enum ApplicationType {
    PoliceOfficerMale("/static/default.jpg", "경찰직공무원(남)"),
    PoliceOfficerFemale("/static/default.jpg", "경찰직공무원(여)"),
    FireOfficerMale("/static/default.jpg", "소방직공무원(남)"),
    FireOfficerFemale("/static/default.jpg", "소방직공무원(여)"),
    CorrectionalOfficerMale("/static/default.jpg", "교정직공무원(남)"),
    CorrectionalOfficerFemale("/static/default.jpg", "교정직공무원(여)");

    private final String iconImageUrl;
    private final String standardName;

    ApplicationType(String iconImageUrl, String standardName) {
        this.iconImageUrl = iconImageUrl;
        this.standardName = standardName;
    }

    public String standardName() {
        return standardName;
    }

    public String iconImageUrl() {
        return iconImageUrl;
    }
}
