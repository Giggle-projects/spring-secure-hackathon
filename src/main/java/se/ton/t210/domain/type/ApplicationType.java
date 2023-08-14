package se.ton.t210.domain.type;

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

    public String getStandardName() {
        return standardName;
    }

    public String iconImageUrl() {
        return iconImageUrl;
    }

    public static String getKeyByName(String name) {
        for (ApplicationType ApplicationType : values()) {
            if (ApplicationType.standardName.equals(name)) {
                return ApplicationType.toString();
            }
        }
        return null;
    }
}
