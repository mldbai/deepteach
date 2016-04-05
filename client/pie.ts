/// <reference path='./typings/browser.d.ts' />
import V = require('virtual-dom')
let {h, diff, patch, create} = V

declare var require: (s: string) => any;
let svg = require('virtual-dom/virtual-hyperscript/svg') as (tagName: string, properties: V.createProperties, children: string | V.VChild[]) =>  V.VNode;

function pie(percent: number, color: string){
    // create a one slice pie chart with percent as percentage and color.
    if( percent < 0.01 ){
        return null
    }
    if( percent > 0.999 ){
        percent = 0.99
    }
    let ra = 10
    let r = -Math.PI * 2 * percent
    let f = (x: number) => Math.round(x * ra)
    let c = svg('path', {
    d : `M0 0 L ${ra} 0 A ${ra} ${ra} 0 ${percent > 0.5 ? 1 : 0 } 0 ${f(Math.cos(r))} ${f(Math.sin(r))} Z `,
        fill: color,
    }, [])
    let t = svg('text', {x: 0, y:0}, percent.toString() )
    let circle = svg('circle', {cx: 0, cy: 0, r: ra+1, stroke: "black", "stroke-width": 1, "fill-opacity": 0.1}, [])
    let g = svg('g', {transform: `translate(${ra+1},${ra+1}) `}, [circle, c])
    return svg('svg', {
        width: 2*ra+2, height: 2*ra+2,
        version : "1.1",
        namespace: "http://www.w3.org/2000/svg",
    }, [g])
}

export = pie
