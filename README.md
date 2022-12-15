<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="https://github.com/junsukha/pose-estimation-demo">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

 <h3 align="center">Anaylyzing Basketball Shooting Poses Using 3D Human Pose Estimation</h3>

  <!-- <p align="center">
    project_description
    <br />
    <a href="https://github.com/junsukha/pose-estimation-demo"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/junsukha/pose-estimation-demo">View Demo</a>
    ·
    <a href="https://github.com/junsukha/pose-estimation-demo/issues">Report Bug</a>
    ·
    <a href="https://github.com/junsukha/pose-estimation-demo/issues">Request Feature</a>
  </p> -->
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

This project analyzes angles of important parts so that users can correct their poses. We use several NBA players' poses as ground truth and compare user's angle with those.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ### Built With

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url] -->

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- GETTING STARTED -->
## Getting Started



### Prerequisites

As you run a file, if an error message complains about something, just install it using pip. For example, if cv2 is not insalled, run

  ```sh
  pip install cv2
  ```

### How to run your video

Run show-video-and-graph.py. Use your video path as an argument. Omit < and > when using argument.

  ```sh
 python show-video-and-graph.py <path_to_your_video>
  ```

Next, run imagesToVideo.py

  ```sh
  python imagesToVideo.py
  ```

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* Siddharth Diwan (HTA)
* Srinath Sridhar (Professor)


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/junsukha/pose-estimation-demo.svg?style=for-the-badge
[contributors-url]: https://github.com/junsukha/pose-estimation-demo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/junsukha/pose-estimation-demo.svg?style=for-the-badge
[forks-url]: https://github.com/junsukha/pose-estimation-demo/network/members
[stars-shield]: https://img.shields.io/github/stars/junsukha/pose-estimation-demo.svg?style=for-the-badge
[stars-url]: https://github.com/junsukha/pose-estimation-demo/stargazers
[issues-shield]: https://img.shields.io/github/issues/junsukha/pose-estimation-demo.svg?style=for-the-badge
[issues-url]: https://github.com/junsukha/pose-estimation-demo/issues
[license-shield]: https://img.shields.io/github/license/junsukha/pose-estimation-demo.svg?style=for-the-badge
[license-url]: https://github.com/junsukha/pose-estimation-demo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 



<!-- This is pose-estimation-demo for Computer Vision final project  
pose_estimation_demo is referred to https://www.analyticsvidhya.com/blog/2021/10/human-pose-estimation-using-machine-learning-in-python/  

The main file is media pipe pose.  

applying-smoothing-curse is base code + smoothing curve function. The output graph is smoothed.  

calculateangle.py calculates angle when a,b,c are given. We are instersted in the angle that ba and bc makes. For example, if a = shoudler position, b = elbow position, c = wrist position, calculateangle.py calculates the angles that elbow-shoulder and elbow-wrist vector makes.   -->

