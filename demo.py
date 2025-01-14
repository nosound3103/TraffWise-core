import utils


def main():
    # utils.detect_objects_with_video(
    #     video_path="data/samples/Traffic Camera VN 4.mp4",
    #     model_path="data/models/detection-2.pt",
    #     output_result_path="data/debug/Traffic Camera VN 4.mp4",
    #     width=1200,
    #     height=900
    # )

    video_path = "vehicles.mp4"

    utils.detect_objects_and_speed_with_video(
        video_path=f"data/samples/{video_path}",
        model_path="data/models/detection-2.pt",
        speed_region=[[1252, 787],
                      [2298, 803],
                      [5039, 2159],
                      [-550, 2159]],
        output_result_path=f"data/debug/{video_path}",
        width=1700,
        height=1200
    )


if __name__ == "__main__":
    main()
